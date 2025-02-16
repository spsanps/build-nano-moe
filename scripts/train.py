# train.py

import time
import torch
import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import wandb

from data import DataLoaderLite, iterate_examples, get_most_likely_row
from GPT import GPT, GPTConfig
from utils import get_lr

def train(
    max_steps=2000,
    warmup_steps=200,
    max_lr=6e-4,
    min_lr=6e-5,
    total_batch_size=524288,
    B=64,
    T=1024,
    device_type='cuda',
    data_root='../data/edu_fineweb10B',
    log_dir='log',
    project_name='my-gpt-project',
    run_name='my-ngpt-run',
    do_hellaswag=False,
    sample_model=False,
    model=None
):
    
    if model is None:
        # default GPT approach
        raise ValueError("No external model provided. Please provide a model to train.")
        config = GPTConfig()
        model = GPT(config)
    else:
        # use the model we passed from main.py
        model = model
        config = model.config
        
    # device = model.device
    
    """
    Main training loop.
    """
    # --- DDP setup
    ddp = int(dist.get_rank()) != -1 if dist.is_initialized() else False
    if ddp:
        ddp_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        # autodetect device if not provided
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    # set seeds for reproducibility
    torch.manual_seed(1337)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(1337)

    # Create data loaders
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="train", data_root=data_root)
    val_loader   = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="val",   data_root=data_root)

    # create model
    # config = GPTConfig()  # example config
    # model = GPT(config)
    # model.to(device)
    raw_model = model  # for logging or checkpoint saving

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module

    # create optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        device_type=device_type,
        master_process=master_process
    )

    # init W&B
    if master_process:
        wandb.init(project=project_name, name=run_name, config={
            "max_steps": max_steps,
            "batch_size": B,
            "seq_len": T,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "grad_accum_steps": grad_accum_steps
        })

    # training loop
    bar = tqdm.tqdm(total=max_steps, disable=not master_process)
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # --- Evaluate val loss occasionally
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            val_loss_accum = 0.0
            val_loss_steps = 20
            with torch.no_grad():
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    val_loss_accum += loss.detach().float()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                val_loss_accum = val_loss_accum / ddp_world_size
            val_loss_accum /= val_loss_steps

            if master_process:
                val_loss_value = val_loss_accum.item()
                print(f"[step {step}] val loss: {val_loss_value:.4f}")
                wandb.log({"val_loss": val_loss_value}, step=step)

        # --- Evaluate HellaSwag (if desired and not compiled)
        if do_hellaswag and (step % 250 == 0 or last_step):
            raise ValueError("HellaSwag implementation is not evaluated yet.")
            num_correct_norm = 0
            num_total = 0
            if master_process:
                print("Evaluating HellaSwag...")

            # Evaluate only on a small subset to keep it quick in practice,
            # or do the full set for a thorough evaluation.
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

            if ddp:
                t_num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                t_num_correct = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(t_num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_num_correct, op=dist.ReduceOp.SUM)
                num_total = t_num_total.item()
                num_correct_norm = t_num_correct.item()

            if master_process:
                acc_norm = num_correct_norm / num_total
                print(f"HellaSwag accuracy: {acc_norm:.4f} ({num_correct_norm}/{num_total})")
                wandb.log({"hellaswag_accuracy": acc_norm}, step=step)

        # --- Optionally sample from the model
        if sample_model and master_process and step > 0 and step % 250 == 0:
            model.eval()
            # (example) top-k sampling from a small prompt
            prompt = "Hello, I'm a language model,"
            num_return_sequences = 2
            max_length = 32
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            xgen = tokens.clone()

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(xgen)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)

            for i in range(num_return_sequences):
                decoded = tokenizer.decode(xgen[i].tolist())
                print(f"[Sample {i}] {decoded}")

        # --- Train step

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # if using DDP, only sync gradients on last micro-step
            if hasattr(model, "require_backward_grad_sync"):
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach().float()
            loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        # Logging
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum = loss_accum / ddp_world_size
        train_loss_val = loss_accum.item()

        dt = time.time() - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            #print(f"step {step} | loss: {train_loss_val:.4f} | lr: {lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            # use progress bar
            bar.set_description(f"loss: {train_loss_val:.4f} | lr: {lr:.4e} | tok/sec: {tokens_per_sec:.2f}")
            bar.update(1)

            wandb.log({"train_loss": train_loss_val, "lr": lr}, step=step)

        # (Optional) save checkpoint
        if master_process and (step % 5000 == 0 or last_step):
            ckpt_path = f"{log_dir}/model_{step:06d}.pt"
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # finalize
    if ddp:
        destroy_process_group()

    if master_process:
        wandb.finish()

    # destroy the progress bar
    if master_process:
        bar.close()

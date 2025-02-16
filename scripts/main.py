import os
import torch
from torch.distributed import init_process_group
import argparse

from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    # existing arguments
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--total_batch_size", type=int, default=131072)
    parser.add_argument("--B", type=int, default=8, help="micro-batch size")
    parser.add_argument("--T", type=int, default=1024, help="sequence length")
    parser.add_argument("--device_type", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--run_name", type=str, default="test-gpt-run")
    parser.add_argument("--do_hellaswag", action='store_true', default=False)
    
    # pick GPT, nGPT or modernGPT
    parser.add_argument("--model_type", type=str, default="gpt",
                        choices=["gpt", "ngpt", "moderngpt"],
                        help="Which model class to use.")
    
    return parser.parse_args()

def main():
    ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if ddp:
        init_process_group(backend='nccl')
    
    args = parse_args()

    # If user picks nGPT or modernGPT, override certain defaults:
    if args.model_type == "ngpt":
        if args.max_lr == 6e-4:
            args.max_lr = 15e-4
        if args.run_name == "test-gpt-run":
            args.run_name = "my-nGPT-run"
    elif args.model_type == "moderngpt":
        # slightly different defaults for a "modern GPT" 
        if args.max_lr == 6e-4:
            pass
            #args.max_lr = 3e-4
        if args.run_name == "test-gpt-run":
            args.run_name = "my-modernGPT-run"

    # Build the model
    if args.model_type == "gpt":
        from GPT import GPT, GPTConfig
        config = GPTConfig(
            block_size=args.T,
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        model = GPT(config)

    elif args.model_type == "ngpt":
        from nGPT import nGPT, nGPTConfig
        config = nGPTConfig(
            block_size=args.T,
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        model = nGPT(config)

    else:  # moderngpt
        from modernGPT import ModernGPT, ModernGPTConfig
        config = ModernGPTConfig(
            block_size=args.T,
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_kv_head=2,  # Multi-Query
            n_embd=768,
            dropout=0.0
        )
        model = ModernGPT(config)

    model.to(args.device_type)
    
    # compile model
    
    model = torch.compile(model)

    train(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        total_batch_size=args.total_batch_size,
        B=args.B,
        T=args.T,
        device_type=args.device_type,
        log_dir=args.log_dir,
        run_name=args.run_name,
        do_hellaswag=args.do_hellaswag,
        sample_model=False,
        model=model  # pass directly to train(...)
    )

if __name__ == "__main__":
    os.makedirs("log", exist_ok=True)
    main()

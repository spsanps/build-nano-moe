# main.py

import os
import torch
from torch.distributed import init_process_group
import argparse

from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=19073)
    parser.add_argument("--warmup_steps", type=int, default=715)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--total_batch_size", type=int, default=131072)
    parser.add_argument("--B", type=int, default=16, help="micro-batch size")
    parser.add_argument("--T", type=int, default=1024, help="sequence length")
    parser.add_argument("--device_type", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--run_name", type=str, default="my-gpt-run")
    parser.add_argument("--do_hellaswag", action='store_true', default=False)
    return parser.parse_args()

def main():
    # Check if we are launched with torchrun for DDP
    ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if ddp:
        init_process_group(backend='nccl')
    
    args = parse_args()

    # call training
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
        do_hellaswag=args.do_hellaswag
    )

if __name__ == "__main__":
    main()

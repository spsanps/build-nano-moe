# build-nano-moe

A collection of popular GPT implementations including a Mixture of Experts (MoE) variant, based on [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt). 

Defaults are set to work well on a Nvidia 4090 GPU (24GB VRAM with bf16 support).

Built with the help of o1-pro. 

## Overview

This repository contains multiple GPT implementations to demonstrate different architectures and optimizations:

1. **GPT**: The original minimal GPT implementation from nanoGPT
2. **ModernGPT**: A modernized GPT with:
   - RMSNorm instead of LayerNorm
   - Rotary Position Embeddings (RoPE)
   - Multi-Query Attention support
   - SwiGLU/Gated MLP options
3. **MoEGPT**: A GPT variant with Mixture of Experts (MoE) based on DeepSeek-MoE:
   - Configurable number of experts
   - Fine-grained expert routing
   - Shared (always active) experts
   - Load balancing loss

## Usage

Train a model for 1000 steps:

```bash
python main.py --model_type [gpt|moderngpt|moegpt] --max_steps 1000 --warmup_steps 100
```

Key arguments:
- `--model_type`: Choose which model implementation to use
- `--max_steps`: Number of training steps
- `--max_lr`: Maximum learning rate
- `--total_batch_size`: Global batch size
- `--B`: Micro-batch size
- `--T`: Sequence length

## Model Details

### MoEGPT

The MoE implementation features:
- `n_experts`: Total number of "macro" experts
- `m_sub_experts`: Sub-experts per macro expert
- `n_shared_experts`: Number of always-active experts
- `n_activated_experts`: Top-k experts to route tokens to
- Load balancing through auxiliary loss

### ModernGPT 

Modern architecture features:
- Multi-Query Attention support (configurable KV heads)
- RoPE positional embeddings
- RMSNorm for stability
- Optional gated or SwiGLU feed-forward
- Flash Attention compatibility

## Requirements

- PyTorch >= 2.0
- transformers (for tokenizer)
- wandb (for logging)

## Acknowledgments

- Base GPT implementation from [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
- MoE design inspired by [DeepSeek-AI's implementation](https://github.com/deepseek-ai/DeepSeek-V3)
- nGPT inspired by [ngpt]( https://github.com/NVIDIA/ngpt)

## License

MIT
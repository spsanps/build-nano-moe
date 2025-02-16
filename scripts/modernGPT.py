import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

################################################################
# Config
################################################################

@dataclass
class ModernGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12        # number of "query" heads
    n_kv_head: int = 1      # number of "key/value" heads for multi-query
    n_embd: int = 768
    dropout: float = 0.0
    use_bias: bool = True
    rope_scaling: float = 10000.0  # base factor for rotary embeddings
    ffn_factor: int = 2           # expansion factor for MLP (SwiGLU etc.)

################################################################
# RoPE utilities
################################################################

def build_sin_cos_rope(seq_len, head_dim, base=10000.0, device=None, dtype=None):
    """
    Create [seq_len, head_dim] of sin/cos curves for RoPE.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    dims = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    # freq ~ 1/(base^(dims/head_dim))
    freqs = positions / (base ** (dims / head_dim))
    sin = freqs.sin()
    cos = freqs.cos()

    rope = torch.zeros((seq_len, head_dim), dtype=dtype, device=device)
    rope[:, 0::2] = sin
    rope[:, 1::2] = cos
    return rope

def apply_rope(x, rope_cache):
    """
    x: [B, n_head, T, head_dim]
    rope_cache: [T, head_dim]
    """
    # rope_cache => (T, head_dim) -> (1,1,T,head_dim)
    sincos = rope_cache.unsqueeze(0).unsqueeze(0)
    sin = sincos[..., 0::2]
    cos = sincos[..., 1::2]

    x0 = x[..., 0::2]
    x1 = x[..., 1::2]

    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin

    out = torch.zeros_like(x)
    out[..., 0::2] = out0
    out[..., 1::2] = out1
    return out

################################################################
# RMSNorm
################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

################################################################
# SwiGLU
################################################################

class SwiGLU(nn.Module):
    """
    Splits input in half: x -> [a|b], then a*silu(b).
    """
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)

################################################################
# MLP
################################################################

class MLP(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        hidden_dim = config.ffn_factor * config.n_embd  # e.g. 2 * n_embd
        # We'll do c_fc => (n_embd -> 2*hidden_dim). Then split => a|b => a*silu(b).
        self.c_fc   = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.use_bias)
        self.act    = SwiGLU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.use_bias)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x

################################################################
# CausalSelfAttention
################################################################

class CausalSelfAttention(nn.Module):
    """
    A 'modern' GPT-style attention with optional multi-query or standard MHA:
    - n_head (Q heads)
    - n_kv_head (K/V heads)
    - Fused Q,K,V in a single linear (c_qkv).
    - If n_kv_head == n_head, we do normal multi-head attention (no replication).
    - Otherwise, if n_kv_head < n_head, replicate K, V to match n_head.
    """

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.n_head_q = config.n_head
        self.n_kv_head = config.n_kv_head
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.head_dim = config.n_embd // config.n_head

        # total = Q + K + V
        # Q dimension = n_head_q * head_dim
        # K dimension = n_kv_head * head_dim
        # V dimension = n_kv_head * head_dim
        # => total_out_dim = (n_head_q + 2*n_kv_head)*head_dim
        total_out_dim = (self.n_head_q + 2 * self.n_kv_head) * self.head_dim

        # single fused linear for Q, K, V
        self.c_qkv = nn.Linear(config.n_embd, total_out_dim, bias=config.use_bias)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.resid_drop = nn.Dropout(config.dropout)

        # scale factor for Q
        self.scale_attn = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, rope_cache=None):
        """
        x: (B, T, n_embd)
        rope_cache: (T, head_dim), optional
        """
        B, T, C = x.shape

        # Fused Q,K,V in one matmul
        qkv = self.c_qkv(x)  # (B, T, total_out_dim)

        # chunk out Q, K, V
        q_out_dim = self.n_head_q * self.head_dim
        kv_out_dim = self.n_kv_head * self.head_dim
        q, k, v = torch.split(qkv, [q_out_dim, kv_out_dim, kv_out_dim], dim=-1)

        # reshape => [B, #heads, T, head_dim]
        q = q.view(B, T, self.n_head_q, self.head_dim).transpose(1, 2)  # [B, n_head_q, T, head_dim]
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # [B, n_kv_head, T, head_dim]
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # [B, n_kv_head, T, head_dim]

        # If n_kv_head == n_head_q, we have standard MHA => no replication needed
        # Otherwise, if n_kv_head < n_head_q, replicate K, V
        if self.n_kv_head < self.n_head_q:
            repeat_factor = self.n_head_q // self.n_kv_head
            k = k.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1)
            k = k.reshape(B, self.n_head_q, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, repeat_factor, -1, -1)
            v = v.reshape(B, self.n_head_q, T, self.head_dim)

        # optional RoPE
        if rope_cache is not None:
            q = apply_rope(q, rope_cache[:T])
            k = apply_rope(k, rope_cache[:T])

        q = q * self.scale_attn

        # scaled dot-product attention (Flash if supported by your GPU / PyTorch)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reassemble => [B, T, n_embd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_drop(y)
        return y

################################################################
# Transformer Block
################################################################

class Block(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope_cache=None):
        x = x + self.attn(self.ln1(x), rope_cache=rope_cache)
        x = x + self.mlp(self.ln2(x))
        return x

################################################################
# ModernGPT (with optimized MHA path)
################################################################

class ModernGPT(nn.Module):
    """
    GPT-like model with:
    - RMSNorm
    - RoPE (no separate wpe)
    - Multi-Query or standard MHA depending on n_kv_head
    - Fused Q,K,V linear
    - SwiGLU MLP
    - Tied embeddings
    """

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

        # build rope cache
        head_dim = config.n_embd // config.n_head
        rope = build_sin_cos_rope(config.block_size, head_dim, base=config.rope_scaling)
        self.register_buffer("rope_cache", rope, persistent=False)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ModernGPT] inited with {n_params/1e6:.2f} M params. n_head={config.n_head}, n_kv_head={config.n_kv_head}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx: (B, T)
        targets: optional (B, T)
        returns: (logits, loss)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence too long."

        x = self.wte(idx)  # token embeddings

        # pass each transformer block
        for block in self.h:
            x = block(x, rope_cache=self.rope_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        if master_process:
            print(f"[ModernGPT] decayed params: {sum(p.numel() for p in decay_params):,}")
            print(f"[ModernGPT] non-decayed params: {sum(p.numel() for p in nodecay_params):,}")

        # Possibly fused
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == "cuda")
        if master_process:
            print(f"[ModernGPT] Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

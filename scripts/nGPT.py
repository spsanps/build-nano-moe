# nGPT.py
# based on https://github.com/NVIDIA/ngpt

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class nGPTConfig:
    # Basic GPT-like configuration
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # nGPT-specific scaling: Typically ~1 / sqrt(n_embd)
    base_scale: float = 1.0 / math.sqrt(768.0)  

###############################################################################
# Helper classes/functions
###############################################################################

class RMSNorm(nn.Module):
    """
    A simple RMSNorm (Root Mean Square LayerNorm).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

def get_sinusoidal_embeddings(n_positions: int, dim: int, device=None):
    """Generate standard sinusoidal [sin, cos] position embeddings."""
    if device is None:
        device = torch.device("cpu")
    position = torch.arange(n_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim), device=device)
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb

def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    """
    Apply rotary embeddings to q, k in-place.
    This expects sinusoidal_pos to be shape [T, head_dim], and q,k to be shape:
      [batch_size, n_heads, seq_len, head_dim].
    """
    # Expand sinusoidal to [1, 1, T, head_dim]
    # but since we do a broadcast, let's do it carefully:
    sincos = sinusoidal_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    # split into sin and cos
    sin, cos = sincos.chunk(2, dim=-1)  # each shape (1,1,T, head_dim/2)

    # for the queries:
    # x[..., ::2], x[..., 1::2] pattern
    # we'll do a direct rearrangement
    def rotary(x):
        # x has shape [b, nH, T, head_dim]
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        # new 0::2 = x0 * cos - x1 * sin
        # new 1::2 = x1 * cos + x0 * sin
        return torch.cat([x0*cos - x1*sin, x1*cos + x0*sin], dim=-1)

    q_out = rotary(q)
    k_out = rotary(k)
    return q_out, k_out

def justnorm(x, dim=-1, eps=1e-8):
    """
    Simple L2 normalization over a given dimension, returning x / ||x||.
    """
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

###############################################################################
# Transformer Block (nGPT style)
###############################################################################

class nGPTBlock(nn.Module):
    """
    One block of the nGPT transformer, combining:
    1) RMSNorm or alpha gating around Self-Attention
    2) RMSNorm or alpha gating around MLP
    3) Rotary Embeddings
    4) Additional gating coefficients:
       - attn_alpha, mlp_alpha, sqk, suv
    """

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        head_dim = config.n_embd // config.n_head

        # Linear transforms for Q, K, V
        self.key   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_out = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # MLP
        self.fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=False)
        self.mlp_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act = nn.SiLU()

        # RMSNorm-like layers
        self.rms_att = RMSNorm(config.n_embd)
        self.rms_mlp = RMSNorm(config.n_embd)

        # nGPT gating parameters
        # (They are scaled versions of alpha gating in the paper.)
        # We'll store them in float32 for safety, then cast as needed
        self.register_parameter(
            "attn_alpha",
            nn.Parameter(torch.ones(config.n_embd, dtype=torch.float32) * (0.05 * config.base_scale))
        )
        self.register_parameter(
            "mlp_alpha",
            nn.Parameter(torch.ones(config.n_embd, dtype=torch.float32) * (0.05 * config.base_scale))
        )
        self.register_parameter(
            "sqk",
            nn.Parameter(torch.ones(config.n_embd, dtype=torch.float32) * (1.0 * config.base_scale))
        )
        self.register_parameter(
            "suv",
            nn.Parameter(torch.ones(2 * 4 * config.n_embd, dtype=torch.float32) * (1.0 * config.base_scale))
        )

    def forward(self, x, sinusoidal_cache=None):
        """
        x: (batch, seq_len, n_embd)
        sinusoidal_cache: precomputed sin/cos for rotary (seq_len, head_dim)
        """
        B, T, C = x.size()
        # 1) Self-Attention
        # Norm
        x_attn_input = self.rms_att(x)     # (B, T, C)
        q = self.query(x_attn_input)
        k = self.key(x_attn_input)
        v = self.value(x_attn_input)

        # Reshape into [B, n_head, T, head_dim]
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Rotary Embeddings
        if sinusoidal_cache is not None:
            q, k = apply_rotary_position_embeddings(sinusoidal_cache, q, k)

        # scale q, k by sqk gating (n_embd float -> shape or broadcast)
        # Typically we interpret sqk as shape [n_embd], but we can broadcast
        # across heads/time. We'll do an approach that normalizes q, k:
        sqk_val = self.sqk * (1.0 / max(self.sqk.abs().max(), 1e-6))
        # direct approach: L2-normalize q, k, then multiply by some factor
        q = justnorm(q, dim=-1) * sqk_val.view(1, 1, 1, -1)[..., :head_dim]
        k = justnorm(k, dim=-1) * sqk_val.view(1, 1, 1, -1)[..., :head_dim]

        # scaled dot-product attention (flash attention if available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # shape [B, n_head, T, head_dim]

        # Recombine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # final projection
        attn_out = self.attn_out(attn_out)

        # alpha gating for residual
        # we L2-normalize the original x, and also L2-normalize attn_out
        # then alpha-blend them
        alpha_attn = self.attn_alpha
        alpha_attn = alpha_attn * (0.05 / max(alpha_attn.abs().max(), 1e-6))  # stable re-scale
        x_norm = justnorm(x, dim=-1)
        y_norm = justnorm(attn_out, dim=-1)
        # simple version: x + alpha*(y - x) => x + alpha*y - alpha*x => (1-alpha)*x + alpha*y
        h_att = x_norm + alpha_attn * (y_norm - x_norm)

        # 2) MLP
        x_mlp_in = self.rms_mlp(h_att)
        uv = self.fc(x_mlp_in)  # shape [B, T, 8*n_embd]
        # apply gating to uv
        # We'll interpret suv similarly to sqk
        suv_val = self.suv * (1.0 / max(self.suv.abs().max(), 1e-6))
        uv = uv * suv_val.view(1, 1, -1)  # broadcast on B,T

        # u, v halves
        u, v = torch.chunk(uv, 2, dim=-1)
        mlp_intermediate = u * self.act(v)

        # project back
        mlp_out = self.mlp_out(mlp_intermediate)

        # alpha gating again
        alpha_mlp = self.mlp_alpha
        alpha_mlp = alpha_mlp * (0.05 / max(alpha_mlp.abs().max(), 1e-6))
        y_norm2 = justnorm(mlp_out, dim=-1)
        h_mlp = h_att + alpha_mlp * (y_norm2 - h_att)

        # L2-normalize final (sometimes done, but you can skip if you want)
        h_final = justnorm(h_mlp, dim=-1)
        return h_final

###############################################################################
# nGPT Model
###############################################################################

class nGPT(nn.Module):
    """
    A GPT-like language model with nGPT gating & RMS norms.
    This has the same signature as the original GPT, so it can be used
    interchangeably in train.py or main.py.
    """

    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.config = config

        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # block modules
        self.blocks = nn.ModuleList([nGPTBlock(config) for _ in range(config.n_layer)])
        # final linear decoder
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        

        # We do NOT tie embeddings in the nGPT paper references 
        # however for better parameter efficiency we do it here
        
        self.lm_head.weight = self.wte.weight

        # build a reusable sinusoidal table for rotary embedding up to config.block_size
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            "sinusoidal_cache",
            get_sinusoidal_embeddings(config.block_size, head_dim),
            persistent=False
        )

        # init weights
        self.apply(self._init_weights)

        print(f"Initialized nGPT model with {self.get_num_params()/1e6:.2f}M parameters.")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        """A simple init that tries to incorporate config.base_scale for normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        """
        idx: (batch, seq_len) of token indices
        targets: (batch, seq_len) of token indices
        Returns: (logits, loss) with shape:
          logits: [batch, seq_len, vocab_size]
          loss: scalar cross-entropy (or None if targets=None)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        # forward the GPT model
        token_emb = self.wte(idx)  # (B, T, n_embd)

        x = token_emb
        # pass through each Transformer block
        for block in self.blocks:
            x = block(x, sinusoidal_cache=self.sinusoidal_cache[:T])

        # only forward final head on each position
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # standard cross entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        """
        Identical signature to GPT. We create an AdamW and return it.
        """
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # matrix/embedding
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # bias etc.
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        if master_process:
            print(f"[nGPT] num decayed params: {sum(p.numel() for p in decay_params):,}")
            print(f"[nGPT] num non-decayed params: {sum(p.numel() for p in nodecay_params):,}")

        # (optional) fused AdamW
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"[nGPT] Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused if use_fused else False
        )
        return optimizer

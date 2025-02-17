# moe.py
# based on: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
# Does not implement Multiheaded Latent Attention (MLA)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

################################################################
# MoeGPTConfig
################################################################

@dataclass
class MoeGPTConfig:
    """
    Minimal GPT config with a MoE feed-forward.
    Adjust defaults to avoid huge param counts.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 8     # reduced from 12 => fewer layers
    n_head: int = 8      # reduced from 12 => 8 heads
    n_embd: int = 512    # reduced from 768 => smaller hidden
    dropout: float = 0.0
    use_bias: bool = True
    rope_scaling: float = 10000.0  # base factor for rotary embeddings
    
    # MoE hyperparams
    # We'll keep only the MoE feed-forward (no standard MLP).
    
    # Fine-grained experts
    n_experts: int = 4         # total "macro" experts
    m_sub_experts: int = 1     # sub-experts per macro => total_experts = n_experts * m_sub_experts
    
    # Shared experts
    n_shared_experts: int = 1  # number of "macro" experts that are always active
                               # => n_always_experts = n_shared_experts*m_sub_experts
    
    # Top-k gating for the other experts
    n_activated_experts: int = 2
    route_score_func: str = "softmax"  # or "sigmoid"
    route_scale: float = 1.0
    
    # MoE feed-forward factor
    ffn_factor: int = 2  # can reduce further if too large
    moe_loss_weight: float = 0.01
    
    # seed
    seed: int = 1337


################################################################
# RoPE utilities
################################################################

def build_sin_cos_rope(seq_len, head_dim, base=10000.0, device=None, dtype=None):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    dims = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
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
    sincos = rope_cache.unsqueeze(0).unsqueeze(0)  # shape [1,1,T,head_dim]
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
# MoEMLP
################################################################

class Expert(nn.Module):
    """
    A single MoE Expert feed-forward:
    [ (W1 & W3) => SwiGLU => W2 ] 
    dimension expansions by ffn_factor
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        hidden_dim = config.ffn_factor * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.use_bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.use_bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.use_bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        # x shape: [num_selected, n_embd]
        a = self.w1(x)
        b = self.w3(x)
        out = F.silu(b) * a
        out = self.w2(out)
        out = self.drop(out)
        return out


class MoEMLP(nn.Module):
    """
    Mixture-of-Experts feed-forward layer:
    - n_total_experts = (n_experts * m_sub_experts)
    - first (n_shared_experts*m_sub_experts) always active
    - top-k chosen among the rest
    - returns (out, bal_loss)
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.dim = config.n_embd

        # total experts
        self.n_macro_experts = config.n_experts
        self.m_sub_experts   = config.m_sub_experts
        self.n_total_experts = self.n_macro_experts * self.m_sub_experts

        # "shared experts" always active
        self.n_shared_macro   = config.n_shared_experts
        self.n_shared         = self.n_shared_macro * self.m_sub_experts

        # top-k among the rest
        self.k = config.n_activated_experts

        # gating linear
        self.score_func = config.route_score_func
        self.route_scale = config.route_scale
        self.gate_linear = nn.Linear(self.dim, self.n_total_experts, bias=True)

        # build experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.n_total_experts)])

        self.moe_loss_weight = config.moe_loss_weight

    def forward(self, x):
        """
        x: (B, T, n_embd)
        returns: (out, balance_loss)
        """
        B, T, C = x.shape
        N = B * T
        xflat = x.reshape(N, C)

        # gating => (N, n_total_experts)
        logits = self.gate_linear(xflat)
        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        else:  # "sigmoid"
            scores = torch.sigmoid(logits)

        # separate "always active" vs. "routable"
        scores_shared    = scores[:, :self.n_shared]           # shape (N, n_shared)
        scores_routable = scores[:, self.n_shared:]           # shape (N, n_total_experts - n_shared)

        topk_vals, topk_idx = torch.topk(scores_routable, k=self.k, dim=-1)
        # offset topk indices
        topk_idx_offset = topk_idx + self.n_shared

        # gather all selected experts
        # shape => (N, n_shared + k)
        shared_ix = torch.arange(self.n_shared, device=x.device).unsqueeze(0).expand(N, self.n_shared)
        all_indices = torch.cat([shared_ix, topk_idx_offset], dim=-1)
        all_weights = torch.cat([scores_shared, topk_vals], dim=-1)
        # scale
        all_weights = all_weights * self.route_scale

        # We'll track usage counts & score sums for each expert
        usage_count = torch.zeros(self.n_total_experts, dtype=torch.long, device=x.device)
        score_sum   = torch.zeros(self.n_total_experts, dtype=scores.dtype, device=x.device)

        # flatten them
        row_ids  = torch.arange(N, device=x.device).unsqueeze(-1).expand(N, self.n_shared + self.k).reshape(-1)
        exp_ids  = all_indices.reshape(-1)
        gate_wts = all_weights.reshape(-1)

        # 1) sort by exp_ids => group tokens that go to the same expert
        sorted_exp_ids, sort_idx = torch.sort(exp_ids)
        sorted_rows   = row_ids[sort_idx]
        sorted_w      = gate_wts[sort_idx]
        sorted_x      = xflat[sorted_rows]  # gather inputs

        # We'll store the forward outputs in a parallel buffer
        outbuf = torch.zeros_like(sorted_x)

        # find boundaries
        boundaries = torch.where(sorted_exp_ids[:-1] != sorted_exp_ids[1:])[0] + 1
        boundaries = torch.cat([
            torch.tensor([0], device=x.device),
            boundaries,
            torch.tensor([len(sorted_exp_ids)], device=x.device)
        ])

        for b in range(len(boundaries)-1):
            start_i = boundaries[b].item()
            end_i   = boundaries[b+1].item()
            e_id    = sorted_exp_ids[start_i].item()

            chunk_x = sorted_x[start_i:end_i]  # input subset
            chunk_w = sorted_w[start_i:end_i]
            # forward
            y = self.experts[e_id](chunk_x)  # shape [num_sel, C]
            # scale
            y = y * chunk_w.unsqueeze(-1)

            outbuf[start_i:end_i] = y

            # usage
            usage_count[e_id] += (end_i - start_i)
            # gating score sum
            score_sum[e_id]   += chunk_w.sum()

        # unsort back to original token order
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(len(sort_idx), device=x.device)
        final_buf = outbuf[inv_sort]  # shape [n_sel, C]

        # 2) scatter_add into out
        # We create zero, do scatter_add_ => no gradient in-place error
        outflat = torch.zeros_like(xflat)
        row_ids_expanded = row_ids.unsqueeze(-1).expand(-1, C)  # shape [N*(n_shared+k), C]
        outflat.scatter_add_(0, row_ids_expanded, final_buf)

        out = outflat.view(B, T, C)

        # ~~~~~ balance loss ~~~~~
        # usage_f = (n_ex / ((n_shared + k)*N)) * usage_count
        # score_p = (1/N) * score_sum
        # L_expbal = sum over i=1..n_ex of usage_f[i]*score_p[i]
        # scaled by self.moe_loss_weight
        n_ex  = float(self.n_total_experts)
        Kp    = float(self.n_shared + self.k)
        usage_f = (n_ex / (Kp*N)) * usage_count.float()
        score_p = (1.0 / N) * score_sum
        bal_loss = (usage_f * score_p).sum()

        return out, bal_loss


################################################################
# CausalSelfAttention (standard MHA)
################################################################

class CausalSelfAttention(nn.Module):
    """
    Standard GPT-like attention:
    - n_head heads
    - n_embd dim total
    - Fused Q,K,V in a single linear
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        total_out_dim = 3 * config.n_embd
        self.c_qkv = nn.Linear(config.n_embd, total_out_dim, bias=config.use_bias)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.resid_drop = nn.Dropout(config.dropout)

        self.scale_attn = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, rope_cache=None):
        B, T, C = x.shape
        qkv = self.c_qkv(x)  # (B, T, 3*C)
        q, k, v = torch.split(qkv, C, dim=-1)

        # reshape => [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,n_head,T,h)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply RoPE if present
        if rope_cache is not None:
            q = apply_rope(q, rope_cache[:T])
            k = apply_rope(k, rope_cache[:T])

        # scaled dot product
        q = q * self.scale_attn
        att = torch.matmul(q, k.transpose(-2, -1))  # (B,n_head,T,T)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        att = att.masked_fill(mask==1, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v)  # (B,n_head,T,h)

        # reassemble
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_drop(out)
        return out


################################################################
# Block
################################################################

class Block(nn.Module):
    """
    GPT block:
    1) x + attn( LN1(x) )
    2) x + MoEMLP( LN2(x) )
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MoEMLP(config)  # always MoE

    def forward(self, x, rope_cache=None):
        # Self-attn
        x = x + self.attn(self.ln1(x), rope_cache=rope_cache)

        # MoE => returns (out, bal_loss)
        out, bal_loss = self.mlp(self.ln2(x))
        x = x + out
        return x, bal_loss


################################################################
# MoeGPT
################################################################

class MoeGPT(nn.Module):
    """
    GPT-like model with an MoEMLP feed-forward layer in each block.
    Also accumulates the balance loss for each block in forward().
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie weights

        # init
        self.apply(self._init_weights)

        # rope cache
        head_dim = config.n_embd // config.n_head
        rope = build_sin_cos_rope(config.block_size, head_dim, base=config.rope_scaling)
        self.register_buffer("rope_cache", rope, persistent=False)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MoeGPT] {n_params/1e6:.2f}M params | "
              f"n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, "
              f"n_experts={config.n_experts}, m_sub_experts={config.m_sub_experts}, shared={config.n_shared_experts}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx: (B, T)
        targets: (B, T) optional
        returns: (logits, loss) if targets is not None, else (logits,)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence too long."

        x = self.wte(idx)

        total_bal_loss = 0.0
        for block in self.blocks:
            x, bal_loss = block(x, rope_cache=self.rope_cache)
            total_bal_loss += bal_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return (logits,)

        # cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
        # add balance loss
        loss = ce_loss + self.config.moe_loss_weight * total_bal_loss
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
            print(f"[MoeGPT] decayed params: {sum(p.numel() for p in decay_params):,}")
            print(f"[MoeGPT] no-decay params: {sum(p.numel() for p in nodecay_params):,}")

        # Possibly fused
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == "cuda")
        if master_process:
            print(f"[MoeGPT] Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

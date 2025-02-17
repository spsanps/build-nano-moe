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
    Minimal GPT config with a Mixture-of-Experts (MoE) feed-forward.
    Adjust defaults to keep parameter sizes smaller for illustration.
    """
    block_size: int = 1024
    vocab_size: int = 50257

    # Transformer architecture
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    use_bias: bool = True

    # Rope scaling (for rotary embeddings)
    rope_scaling: float = 10000.0

    # MoE hyperparams
    n_experts: int = 4          # total "macro-experts"
    m_sub_experts: int = 1       # sub-experts per macro => total = n_experts*m_sub_experts
    n_shared_experts: int = 1    # how many "macro" experts are always active
    n_activated_experts: int = 2 # top-k gating among the rest

    route_score_func: str = "softmax"  # or "sigmoid"
    route_scale: float = 1.0           # multiply gating scores for stability
    ffn_factor: int = 2                # multiplier for hidden dim in the feed-forward
    moe_loss_weight: float = 0.01      # alpha for MoE balance loss

    # Which MoEMLP implementation
    #   "chunk":    chunk-based gating + explicit Expert modules
    #   "parallel": single parameter set for all experts (default and recommended)
    moe_impl: str = "parallel"

    # Random seed (for reproducibility if needed)
    seed: int = 1337


################################################################
# Rope utilities
################################################################

def build_sin_cos_rope(seq_len, head_dim, base=10000.0, device=None, dtype=None):
    """
    Creates [seq_len, head_dim] containing interleaved sin/cos frequencies
    for rotary position embeddings (RoPE).

    - seq_len: maximum sequence length
    - head_dim: dimension per attention head
    - base: base frequency for RoPE
    """
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
    Apply rotary position embeddings (RoPE) to Q,K.

    x:          [B, n_head, T, head_dim]
    rope_cache: [T, head_dim]
    """
    # Expand rope_cache from [T, head_dim] -> [1,1,T,head_dim]
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
    """
    A simple RMSNorm: out = weight * x / sqrt(mean(x^2) + eps)
    """
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
# CausalSelfAttention (fused Q,K,V)
################################################################

class CausalSelfAttention(nn.Module):
    """
    Standard GPT-like multi-head attention, with:
    - Single fused Q,K,V linear
    - PyTorch 2.x scaled_dot_product_attention for efficiency
    - Output projection + optional dropout
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # single fused Q,K,V => out_dim = 3 * n_embd
        self.c_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.use_bias)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, rope_cache=None):
        """
        x: (B, T, n_embd)
        rope_cache: (T, head_dim) for RoPE
        """
        B, T, C = x.shape
        qkv = self.c_qkv(x)  # (B, T, 3*C)
        q, k, v = torch.split(qkv, C, dim=-1)

        # reshape => [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply RoPE to Q,K
        if rope_cache is not None:
            q = apply_rope(q, rope_cache[:T])
            k = apply_rope(k, rope_cache[:T])

        # scaled dot-product attention (is_causal=True)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reassemble => [B, T, n_embd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_drop(y)
        return y

################################################################
# MoEMLP Implementation #1 (Chunk-based)
################################################################

class Expert(nn.Module):
    """
    A single MoE Expert feed-forward for the chunk-based approach:
      out = W2( a * silu(b) ),
    where [a,b] = chunk( c_fc(x), 2 )
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        hidden_dim = config.ffn_factor * config.n_embd
        # Fused W1 & W3 => shape: (n_embd -> 2*hidden_dim)
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.use_bias)
        self.w2   = nn.Linear(hidden_dim, config.n_embd, bias=config.use_bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [num_tokens_assigned, n_embd]
        fc_out = self.c_fc(x)  # => [num_tokens_assigned, 2*hidden_dim]
        a, b = fc_out.chunk(2, dim=-1)
        ab = a * F.silu(b)
        out = self.w2(ab)
        out = self.drop(out)
        return out


class MoEMLPChunk(nn.Module):
    """
    The "chunk-based" MoE MLP:
      - We compute gating scores => top-k + shared
      - Group tokens by their assigned experts, run each Expert on that chunk
      - Scatter-add results back
      - Compute balance loss using sum of *post-activation* gating probabilities
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.dim = config.n_embd

        # total experts
        self.n_macro_experts = config.n_experts
        self.m_sub_experts   = config.m_sub_experts
        self.n_total_experts = self.n_macro_experts * self.m_sub_experts

        # "shared" experts always active
        self.n_shared_macro = config.n_shared_experts
        self.n_shared = self.n_shared_macro * self.m_sub_experts

        # top-k among the rest
        self.k = config.n_activated_experts

        # gating linear
        self.score_func = config.route_score_func
        self.route_scale = config.route_scale
        self.gate_linear = nn.Linear(self.dim, self.n_total_experts, bias=True)

        # build separate Expert modules
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.n_total_experts)])
        self.moe_loss_weight = config.moe_loss_weight

    def forward(self, x):
        """
        x: (B, T, n_embd)
        returns: (out, bal_loss)
        """
        B, T, C = x.shape
        N = B * T
        xflat = x.reshape(N, C)

        # 1) gating => (N, n_total_experts)
        logits = self.gate_linear(xflat)

        # 2) post-activation probabilities => ensures non-negative values
        if self.score_func == "softmax":
            probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        else:  # "sigmoid"
            probs = torch.sigmoid(logits)

        # 3) sum of these probabilities => used for P_i in balance loss
        #    (we no longer sum raw logits, which can be negative!)
        score_sum = probs.sum(dim=0)  # shape(n_total_experts,)

        # 4) separate "always active" vs. "routable"
        probs_shared    = probs[:, :self.n_shared]         # shape (N, n_shared)
        probs_routable = probs[:, self.n_shared:]          # shape (N, n_total_experts - n_shared)

        # top-k among routable
        topk_vals, topk_idx = torch.topk(probs_routable, k=self.k, dim=-1)
        topk_idx_offset = topk_idx + self.n_shared

        # combine shared + topk
        shared_ix = torch.arange(self.n_shared, device=x.device).unsqueeze(0).expand(N, self.n_shared)
        all_indices = torch.cat([shared_ix, topk_idx_offset], dim=-1)  # shape(N, n_shared+k)
        all_weights = torch.cat([probs_shared, topk_vals], dim=-1)
        all_weights = all_weights * self.route_scale  # optional gating scale

        # Flatten for chunk-based routing
        row_ids  = torch.arange(N, device=x.device).unsqueeze(-1).expand(N, self.n_shared + self.k).reshape(-1)
        exp_ids  = all_indices.reshape(-1)
        gate_wts = all_weights.reshape(-1)

        # Sort by exp_id => group tokens for each expert
        sorted_exp_ids, sort_idx = torch.sort(exp_ids)
        sorted_rows   = row_ids[sort_idx]
        sorted_w      = gate_wts[sort_idx]
        sorted_x      = xflat[sorted_rows]

        # We'll accumulate partial outputs in outbuf (same shape as sorted_x)
        outbuf = torch.zeros_like(sorted_x)

        # Boundaries between different experts
        boundaries = torch.where(sorted_exp_ids[:-1] != sorted_exp_ids[1:])[0] + 1
        boundaries = torch.cat([
            torch.tensor([0], device=x.device),
            boundaries,
            torch.tensor([len(sorted_exp_ids)], device=x.device)
        ])

        # usage count (# of tokens assigned) for usage_factor
        usage_count = torch.zeros(self.n_total_experts, dtype=torch.long, device=x.device)

        # For each expert chunk => forward pass
        for b in range(len(boundaries) - 1):
            start_i = boundaries[b].item()
            end_i   = boundaries[b+1].item()
            e_id    = sorted_exp_ids[start_i].item()

            chunk_x = sorted_x[start_i:end_i]
            chunk_w = sorted_w[start_i:end_i]

            # forward pass
            y = self.experts[e_id](chunk_x)  # shape [num_sel, C]
            # scale by gating weights
            y = y * chunk_w.unsqueeze(-1)

            outbuf[start_i:end_i] = y
            usage_count[e_id] += (end_i - start_i)

        # unsort back
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(len(sort_idx), device=x.device)
        final_buf = outbuf[inv_sort]

        # scatter-add to get final
        outflat = torch.zeros_like(xflat)
        row_ids_expanded = row_ids.unsqueeze(-1).expand(-1, C)
        outflat.scatter_add_(0, row_ids_expanded, final_buf)
        out = outflat.view(B, T, C)

        # ~~~~~~~~~ Balance Loss ~~~~~~~~~
        # usage_factor f_i = (N_ex / (K' * N)) * usage_count[i]
        # P_i = (1.0 / N) * score_sum[i]    (sum of post-scores across all tokens)
        # K' = n_shared + k
        Kp = float(self.n_shared + self.k)
        Np = float(self.n_total_experts)
        usage_factor = (Np / (Kp * N)) * usage_count.float()
        score_prob   = (1.0 / N) * score_sum
        bal_loss = (usage_factor * score_prob).sum()

        return out, bal_loss


################################################################
# MoEMLP Implementation #2 (Single-parameter parallel approach)
################################################################

class MoEMLPParallel(nn.Module):
    """
    Parallel Mixture-of-Experts feed-forward with top-K gating.
    Uses a single large parameter for all experts (improves efficiency).

    Steps:
      1) gating logits => post-activation (softmax or sigmoid)
      2) pick shared + top-K => gating
      3) big parallel matmul for all experts
      4) weighting + sum
      5) compute usage_factor & score_prob for balance loss
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.dim = config.n_embd

        # total experts
        self.n_macro_experts = config.n_experts
        self.m_sub_experts   = config.m_sub_experts
        self.n_total_experts = self.n_macro_experts * self.m_sub_experts

        # "shared" experts
        self.n_shared_macro = config.n_shared_experts
        self.n_shared = self.n_shared_macro * self.m_sub_experts

        # top-k gating
        self.k = config.n_activated_experts

        self.score_func = config.route_score_func
        self.route_scale = config.route_scale
        self.moe_loss_weight = config.moe_loss_weight

        # dimension for feed-forward hidden
        self.hidden_dim = config.ffn_factor * config.n_embd

        # gating linear => (n_embd -> n_total_experts)
        self.gate_linear = nn.Linear(self.dim, self.n_total_experts, bias=True)

        # Single large parameter set:
        #   fc_weight   => (n_experts, 2*hidden_dim, n_embd)
        #   fc_bias     => (n_experts, 2*hidden_dim)
        #   proj_weight => (n_experts, hidden_dim, n_embd)
        #   proj_bias   => (n_experts, n_embd)
        self.fc_weight    = nn.Parameter(torch.zeros(self.n_total_experts, 2*self.hidden_dim, self.dim))
        self.fc_bias      = nn.Parameter(torch.zeros(self.n_total_experts, 2*self.hidden_dim))
        self.proj_weight  = nn.Parameter(torch.zeros(self.n_total_experts, self.hidden_dim, self.dim))
        self.proj_bias    = nn.Parameter(torch.zeros(self.n_total_experts, self.dim))

        self.drop = nn.Dropout(config.dropout)

        # init
        nn.init.normal_(self.fc_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_bias)
        nn.init.normal_(self.proj_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_bias)

    def forward(self, x):
        """
        x: (B, T, n_embd)
        returns: (out, bal_loss)
        """
        B, T, C = x.shape
        N = B * T
        xflat = x.view(N, C)

        # 1) gating logits => shape (N, n_experts)
        logits = self.gate_linear(xflat)

        # 2) post-activation probabilities (non-negative)
        if self.score_func == "softmax":
            post_scores = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        else:
            post_scores = torch.sigmoid(logits)

        # sum of probabilities => used for balance loss
        score_sum = post_scores.sum(dim=0)  # shape(n_experts,)

        # 3) separate "shared" vs "routable"
        shared_part = post_scores[:, :self.n_shared]      # (N, n_shared)
        rest_part   = post_scores[:, self.n_shared:]      # (N, n_experts - n_shared)

        # top-k from the "rest"
        topk_vals, topk_idx = torch.topk(rest_part, k=self.k, dim=-1)
        # create a zero mask
        mask = torch.zeros_like(rest_part)
        mask.scatter_(1, topk_idx, topk_vals)

        # combine shared + mask
        combined_scores = torch.cat([shared_part, mask], dim=-1)  # shape(N, n_experts)
        combined_scores = combined_scores * self.route_scale

        # normalize among selected => final gating distribution
        gating_denom = combined_scores.sum(dim=-1, keepdim=True) + 1e-8
        gating = combined_scores / gating_denom  # shape(N, n_experts)

        # usage factor => how many tokens went to each expert
        selected_mask = (combined_scores > 0).float()
        usage_count = selected_mask.sum(dim=0)  # shape(n_experts,)

        # 4) big parallel forward for all experts
        # fc_out => (N,e,2h)
        fc_out = torch.einsum('nd,ecd->nec', xflat, self.fc_weight)
        fc_out = fc_out + self.fc_bias.unsqueeze(0)  # broadcast bias => shape(1,e,2h)

        a, b = fc_out.split(self.hidden_dim, dim=-1)  # => (N,e,h)
        hidden = a * F.silu(b)

        # out_experts => (N,e,c)
        out_experts = torch.einsum('neh,ehc->nec', hidden, self.proj_weight)
        out_experts = out_experts + self.proj_bias.unsqueeze(0)
        out_experts = self.drop(out_experts)

        # Weighted sum => gating => shape(N,e)
        out = (out_experts * gating.unsqueeze(-1)).sum(dim=1)  # => (N,c)
        out = out.view(B, T, C)

        # 5) Balance loss => sum_i [ f_i * P_i ]
        #    where f_i = (n_experts / (K' * N)) * usage_count[i]
        #          P_i = (1.0 / N) * score_sum[i]
        #    K' = n_shared + k
        Kp = float(self.n_shared + self.k)
        Np = float(self.n_total_experts)

        usage_factor = (Np / (Kp * N)) * usage_count.float()
        score_prob   = (1.0 / N) * score_sum
        bal_loss = (usage_factor * score_prob).sum()

        return out, bal_loss


################################################################
# Unified MoEMLP: pick chunk-based or parallel
################################################################

class MoEMLP(nn.Module):
    """
    Wraps either:
      - MoEMLPChunk (moe_impl='chunk')
      - MoEMLPParallel (moe_impl='parallel')
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.moe_impl = config.moe_impl.lower()
        if self.moe_impl == "chunk":
            self.impl = MoEMLPChunk(config)
        else:
            self.impl = MoEMLPParallel(config)

    def forward(self, x):
        return self.impl(x)


################################################################
# Block
################################################################

class Block(nn.Module):
    """
    A GPT block with:
      1) x + attn( LN1(x) )
      2) x + MoEMLP( LN2(x) ) => mixture-of-experts feed-forward
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MoEMLP(config)

    def forward(self, x, rope_cache=None):
        """
        x: (B, T, n_embd)
        rope_cache: (T, head_dim) for RoPE
        Returns: (x_out, bal_loss)
        """
        # 1) self-attention sub-layer
        x = x + self.attn(self.ln1(x), rope_cache=rope_cache)

        # 2) MoE feed-forward sub-layer
        mlp_out, bal_loss = self.mlp(self.ln2(x))
        x = x + mlp_out

        return x, bal_loss


################################################################
# MoeGPT
################################################################

class MoeGPT(nn.Module):
    """
    GPT-like model with:
      - Input embedding
      - n_layer blocks (Attn + MoEMLP)
      - Final RMSNorm
      - LM head (ties weights with the embedding)
      - Summed "balance loss" from each block
    """
    def __init__(self, config: MoeGPTConfig):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config

        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)

        # Output (LM) head -- tie with wte
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        # Initialize parameters
        self.apply(self._init_weights)

        # rope cache
        head_dim = config.n_embd // config.n_head
        rope = build_sin_cos_rope(config.block_size, head_dim, base=config.rope_scaling)
        self.register_buffer("rope_cache", rope, persistent=False)

        # Print model info
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MoeGPT] {n_params/1e6:.2f}M params | "
              f"n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, "
              f"n_experts={config.n_experts}, m_sub_experts={config.m_sub_experts}, "
              f"moe_impl={config.moe_impl}")

    def _init_weights(self, module):
        """
        Simple weight initialization:
          - Normal(0, 0.02) for all Linear/Embedding weights
          - Zeros for all biases
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass:
          idx: (B, T) token indices
          targets: (B, T) optional
        Returns:
          if targets is None: (logits,)
          else: (logits, { 'loss': total_loss, 'ce_loss': ce_loss, 'moe_loss': sum_of_balance_losses })
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence too long."

        # 1) token embedding
        x = self.wte(idx)

        # 2) pass through each Transformer block
        total_bal_loss = 0.0
        for block in self.blocks:
            x, bal_loss = block(x, rope_cache=self.rope_cache)
            total_bal_loss += bal_loss

        # 3) final RMSNorm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # If no targets, just return the logits
        if targets is None:
            return (logits,)

        # ~~~~ Compute CE loss ~~~~
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )

        # ~~~~ Total loss = CE + alpha * sum(bal_loss) ~~~~
        loss = ce_loss + self.config.moe_loss_weight * total_bal_loss

        # Return dictionary with separate components
        loss_dict = {
            'loss': loss,
            'ce_loss': ce_loss,
            'moe_loss': total_bal_loss
        }
        return logits, loss_dict

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        """
        Returns an AdamW optimizer with optional fused variant (PyTorch 2.0+ on GPU).
        Typically used in a training loop external to this file.
        """
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

        # Possibly use fused AdamW if available and on GPU
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

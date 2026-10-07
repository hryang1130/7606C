"""Transformer noise predictor migrated from VariDP.

Ported from ``VariDP/dp/backbones.py`` (``TransformerForDiffusion``), which
follows ``real-stanford/diffusion_policy``'s ``transformer_for_diffusion.py``
with the paper's DP-T lowdim configuration: a causal decoder over action tokens
attends to ``[time token ‖ observation tokens]`` memory, using 8 layers,
4 heads, 256 embeddings and attention dropout 0.3.

The network is kept faithful to the donor. The only layout adaptation is that
the sinusoidal time embedding comes from ``timestep.py`` (shared with the MLP
backbone) instead of being redefined here. ``TransformerBackbone`` adapts the
network to the shared ``NoisePredictor`` contract, exactly like
``UNetBackbone`` does for ``ConditionalUnet1D``.
"""

from __future__ import annotations

import importlib.util
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PolicyConfig
from .base import NoisePredictor
from .timestep import SinusoidalPosEmb


def _manual_attention_enabled() -> bool:
    """Whether to bypass the fused ``scaled_dot_product_attention`` kernel.

    Moore Threads (MUSA) builds cannot use ``torch_musa``'s fp32 SDPA backward:
    gradients spike to 1e6..1e11, ``grad_clip`` then locks the model, and the
    diffusion loss stalls above 1.0 instead of decreasing (observed on
    PickCube/StackCube transformer runs).  The unet/mlp backbones contain no
    attention and are unaffected, which is why only this backbone fails there.

    Override with ``DP_TRANSFORMER_ATTN=manual`` (always manual) or
    ``DP_TRANSFORMER_ATTN=sdpa`` (always the fused kernel, e.g. on CUDA).
    """
    mode = os.environ.get("DP_TRANSFORMER_ATTN", "auto").strip().lower()
    if mode == "manual":
        return True
    if mode == "sdpa":
        return False
    for name in ("torchada", "torch_musa"):
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


class ManualMultiheadAttention(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention`` without the fused kernel.

    Computes ``softmax(q kᵀ / sqrt(d)) v`` explicitly, reusing the very same
    ``Parameter`` objects as the module it replaces, so state-dict names,
    initialisation and optimizer param groups are unchanged.
    """

    def __init__(self, src: nn.MultiheadAttention) -> None:
        super().__init__()
        self.embed_dim = src.embed_dim
        self.num_heads = src.num_heads
        self.head_dim = src.head_dim
        self.dropout = src.dropout
        self.batch_first = True
        self._qkv_same_embed_dim = src._qkv_same_embed_dim
        if self._qkv_same_embed_dim:
            self.in_proj_weight = src.in_proj_weight
        else:
            self.q_proj_weight = src.q_proj_weight
            self.k_proj_weight = src.k_proj_weight
            self.v_proj_weight = src.v_proj_weight
        self.in_proj_bias = src.in_proj_bias
        self.bias_k = src.bias_k
        self.bias_v = src.bias_v
        self.add_zero_attn = src.add_zero_attn
        self.out_proj = src.out_proj

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        bsz, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]

        if self._qkv_same_embed_dim:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        else:
            w_q, w_k, w_v = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
        b_q = b_k = b_v = None
        if self.in_proj_bias is not None:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        if self.bias_k is not None:                      # pragma: no cover (unused here)
            k = torch.cat([k, self.bias_k.expand(bsz, -1, -1)], dim=1)
            v = torch.cat([v, self.bias_v.expand(bsz, -1, -1)], dim=1)
            src_len += 1

        heads, head_dim = self.num_heads, self.head_dim
        q = q.reshape(bsz, tgt_len, heads, head_dim).transpose(1, 2)
        k = k.reshape(bsz, src_len, heads, head_dim).transpose(1, 2)
        v = v.reshape(bsz, src_len, heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if is_causal and attn_mask is None:
            causal = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)                       # (B, H, L, D)
        out = out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None


class TransformerForDiffusion(nn.Module):
    def __init__(self, input_dim, output_dim, horizon, n_obs_steps=None, cond_dim=0,
                 n_layer: int = 12, n_head: int = 12, n_emb: int = 768,
                 p_drop_emb: float = 0.1, p_drop_attn: float = 0.1,
                 causal_attn: bool = False, time_as_cond: bool = True,
                 obs_as_cond: bool = False, n_cond_layers: int = 0):
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True      # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s - 1)   # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)

        # MUSA (Moore Threads) safety: swap the fused-SDPA attention out before
        # the module is ever moved to a device. No-op on CUDA builds.
        self.install_manual_attention_if_needed()

    def install_manual_attention_if_needed(self) -> bool:
        """Replace encoder/decoder attention with the explicit matmul/softmax path.

        Returns True when the swap happened. Weights are shared, so this is
        numerically identical to ``nn.MultiheadAttention`` in exact arithmetic;
        it only avoids the fused kernel whose MUSA fp32 backward is broken.
        """
        if not _manual_attention_enabled():
            return False
        layers = [m for m in self.modules()
                  if isinstance(m, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))]
        swapped = 0
        for module in layers:
            if isinstance(module, nn.TransformerEncoderLayer):
                module.self_attn = ManualMultiheadAttention(module.self_attn)
                swapped += 1
            else:
                module.self_attn = ManualMultiheadAttention(module.self_attn)
                module.multihead_attn = ManualMultiheadAttention(module.multihead_attn)
                swapped += 2
        self.manual_attention_layers = swapped
        return swapped > 0

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential,
                        ManualMultiheadAttention)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self, sample, timestep, cond=None, **kwargs):
        """
        sample   : (B, T, input_dim)    action sequence to denoise
        timestep : (B,) long / int      raw diffusion step
        cond     : (B, To, cond_dim)    per-step observation condition tokens
        output   : (B, T, output_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)

            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


class TransformerBackbone(NoisePredictor):
    """Keep ``(B, To, Dobs)`` as condition tokens for the migrated DP-T.

    Structure comes from ``PolicyConfig`` (``transformer_*`` fields in
    ``baseline.toml``, donor defaults: 8 layers, 4 heads, 256 embeddings, causal
    attention). The shared observation sequence is consumed unchanged; this
    adapter never re-encodes RGB or proprioception.
    """

    def __init__(self, policy_cfg: PolicyConfig, *, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_horizon = policy_cfg.obs_horizon
        self.transformer = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=policy_cfg.pred_horizon,
            n_obs_steps=policy_cfg.obs_horizon,
            cond_dim=obs_dim,
            n_layer=policy_cfg.transformer_layers,
            n_head=policy_cfg.transformer_heads,
            n_emb=policy_cfg.transformer_embed_dim,
            p_drop_emb=policy_cfg.transformer_dropout_emb,
            p_drop_attn=policy_cfg.transformer_dropout_attn,
            causal_attn=policy_cfg.transformer_causal_attn,
            time_as_cond=True,
            n_cond_layers=policy_cfg.transformer_cond_layers,
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        if obs_features.ndim != 3:
            raise ValueError(
                f"expected observation features (B, To, Dobs), got {tuple(obs_features.shape)}"
            )
        return self.transformer(noisy_actions, timestep, cond=obs_features)

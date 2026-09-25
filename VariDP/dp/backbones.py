"""backbones.py — DP 的三种噪声预测主干

统一约定：
  * mlp / transformer：forward(x, t_emb, c) -> (B, Tp, A)
      x (B, Tp, A) 待去噪动作序列，t_emb (B, 128) 预计算时间嵌入，c (B, cond_dim) 条件
  * unet：官方 ConditionalUnet1D 的完整移植（签名与官方一致）
      forward(sample, timestep, local_cond=None, global_cond=None) -> (B, Tp, A)
      接收原始 timestep，时间嵌入与 global_cond 在主干内部拼接后再做 FiLM；
      DiffusionPolicy.eps() 按 backbone 名分发到对应签名。
"""
import math

import torch
import torch.nn as nn


class MLPNoisePred(nn.Module):
    """原 dp_lib 的实现：展平拼接 -> MLP（baseline，行为与旧版完全一致）"""

    def __init__(self, Tp, act_dim, cond_dim, t_dim=128, hidden=256, n_layers=3):
        super().__init__()
        self.Tp, self.act_dim = Tp, act_dim
        dims = [Tp * act_dim + t_dim + cond_dim] + [hidden] * n_layers + [Tp * act_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.Mish(), nn.LayerNorm(dims[i + 1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, c):
        b = x.shape[0]
        return self.net(torch.cat([x.reshape(b, -1), t, c], dim=-1)
                        ).reshape(b, self.Tp, self.act_dim)


# --------------------------------------------------------------------------- #
# UNet 主干：逐行对齐 real-stanford/diffusion_policy（论文 state / lowdim 配置）
#   diffusion_policy/model/diffusion/conv1d_components.py
#   diffusion_policy/model/diffusion/conditional_unet1d.py
#   diffusion_policy/model/diffusion/positional_embedding.py
#   超参取 config/train_diffusion_unet_lowdim_workspace.yaml：
#     diffusion_step_embed_dim=256, down_dims=[256,512,1024],
#     kernel_size=5, n_groups=8, cond_predict_scale=True
#   与官方唯一的实现差异：官方用 einops.Rearrange，这里用等价的 unsqueeze（不引入新依赖）
# --------------------------------------------------------------------------- #
class SinusoidalPosEmb(nn.Module):
    """官方 positional_embedding.py：sin/cos 时间步嵌入（dim = diffusion_step_embed_dim）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """官方 conv1d_components.py：Conv1d -> GroupNorm -> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    """官方 conv1d_components.py：Conv1d(dim, dim, 3, stride=2, padding=1)"""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """官方 conv1d_components.py：ConvTranspose1d(dim, dim, 4, 2, 1)（可学习上采样）"""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """官方 ConditionalResidualBlock1D

    两个 Conv1dBlock + FiLM 条件调制；cond 为 [时间嵌入 ‖ global_cond] 拼接后的向量。
    cond_predict_scale=True 时预测 per-channel scale 与 bias（论文 lowdim 配置即为 True）。
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3,
                 n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x    : [B, in_channels, horizon]
        cond : [B, cond_dim]
        out  : [B, out_channels, horizon]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)[:, :, None]   # 等价 Rearrange('batch t -> batch t 1')
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale, bias = embed[:, 0, ...], embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """官方 ConditionalUnet1D（论文 state/lowdim 版：1D 时序 UNet + FiLM）

    接口与官方一致：
        forward(sample, timestep, local_cond=None, global_cond=None)
            sample      : (B, Tp, input_dim)   待去噪的动作序列
            timestep    : (B,) long / int      原始扩散步（主干内部自己做时间嵌入）
            global_cond : (B, global_cond_dim) 观测条件（本项目 = 归一化 obs 展平）
        return          : (B, Tp, input_dim)
    """

    def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8, cond_predict_scale=True):
        super().__init__()
        down_dims = list(down_dims)
        all_dims = [input_dim] + down_dims
        start_dim = down_dims[0]

        # 时间步编码 + MLP（官方 diffusion_step_encoder）
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # 条件维度 = 时间嵌入 + global_cond（官方：单一条件通路）
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size,
                n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size,
                n_groups=n_groups, cond_predict_scale=cond_predict_scale),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)   # 官方原样：3 层时恒为 False，故每级都上采样
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size,
                    n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample, timestep, local_cond=None, global_cond=None, **kwargs):
        sample = sample.transpose(1, 2)                # (B, Tp, D) -> (B, D, Tp)

        # 1. 时间步嵌入；官方做法：把时间嵌入与 global_cond 拼成一个条件向量
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # 2. 局部条件编码（本项目未用 local cond，保留官方分支以便逐行对照）
        h_local = list()
        if local_cond is not None:
            local_cond = local_cond.transpose(1, 2)     # (B, Tp, D) -> (B, D, Tp)
            resnet, resnet2 = self.local_cond_encoder
            x_local = resnet(local_cond, global_feature)
            h_local.append(x_local)
            x_local = resnet2(local_cond, global_feature)
            h_local.append(x_local)

        # 3. 下采样（每级 2 个残差块 + skip 缓存，最后一级不下采样）
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # 4. 中间层（2 个残差块）
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # 5. 上采样（concat skip -> 2 个残差块 -> 上采样）
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # 官方原样保留（该条件恒不成立，官方注释说明改了会破坏已发布 checkpoint）
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)                       # (B, D, Tp) -> (B, Tp, D)


# --------------------------------------------------------------------------- #
# Transformer 主干：逐行对齐 real-stanford/diffusion_policy（论文 DP-T）
#   diffusion_policy/model/diffusion/transformer_for_diffusion.py
#   超参取 config/train_diffusion_transformer_lowdim_workspace.yaml：
#     n_layer=8, n_head=4, n_emb=256, p_drop_emb=0.0, p_drop_attn=0.3,
#     causal_attn=True, time_as_cond=True, obs_as_cond=True, n_cond_layers=0
#   结构：cond = [时间 token ‖ 逐步 obs token] -> cond encoder(MLP) -> memory；
#         动作 token + 可学习位置编码 -> TransformerDecoder（因果 mask + cross-attn）-> head
#   与官方唯一的实现差异：不继承 ModuleAttrMixin（本项目的 device/dtype 由 train/eval 管），
#                         也未移植官方的 configure_optimizers（本项目统一 AdamW）
# --------------------------------------------------------------------------- #
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

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
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
        sample   : (B, T, input_dim)   待去噪动作序列
        timestep : (B,) long / int     原始扩散步（主干内部做 SinusoidalPosEmb）
        cond     : (B, To, cond_dim)   逐步观测条件（本项目 = 归一化 obs[:, :To]）
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


BACKBONES = ("mlp", "unet", "transformer")


def build_noise_pred(backbone, Tp, act_dim, cond_dim, t_dim=128,
                     hidden=256, n_layers=3,                  # 仅 mlp 用
                     unet_down_dims=(256, 512, 1024),         # 仅 unet 用（论文 lowdim 配置）
                     unet_kernel_size=5, unet_n_groups=8,
                     unet_step_embed_dim=256, unet_cond_predict_scale=True,
                     n_obs_steps=2,                                     # 仅 transformer 用
                     tf_n_layer=8, tf_n_head=4, tf_n_emb=256,           # 仅 transformer 用
                     tf_p_drop_emb=0.0, tf_p_drop_attn=0.3,
                     tf_causal_attn=True, tf_n_cond_layers=0):
    """按名字构造噪声预测主干（unet / transformer 均为官方实现的完整移植）"""
    if backbone == "mlp":
        return MLPNoisePred(Tp, act_dim, cond_dim, t_dim,
                            hidden=hidden, n_layers=n_layers)
    if backbone == "unet":
        down_dims = list(unet_down_dims)
        n_down = len(down_dims) - 1
        if Tp % (2 ** n_down) != 0:
            raise ValueError(
                f"Tp={Tp} 必须能被 2^(len(down_dims)-1)={2 ** n_down} 整除"
                f"（down_dims={down_dims}）")
        return ConditionalUnet1D(
            input_dim=act_dim, local_cond_dim=None, global_cond_dim=cond_dim,
            diffusion_step_embed_dim=unet_step_embed_dim, down_dims=down_dims,
            kernel_size=unet_kernel_size, n_groups=unet_n_groups,
            cond_predict_scale=unet_cond_predict_scale)
    if backbone == "transformer":
        return TransformerForDiffusion(
            input_dim=act_dim, output_dim=act_dim, horizon=Tp,
            n_obs_steps=n_obs_steps, cond_dim=cond_dim,
            n_layer=tf_n_layer, n_head=tf_n_head, n_emb=tf_n_emb,
            p_drop_emb=tf_p_drop_emb, p_drop_attn=tf_p_drop_attn,
            causal_attn=tf_causal_attn, time_as_cond=True,
            obs_as_cond=cond_dim > 0, n_cond_layers=tf_n_cond_layers)
    raise ValueError(f"未知 backbone: {backbone}，可选 {BACKBONES}")

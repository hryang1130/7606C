# dp_manip

State-based Diffusion Policy training and evaluation for ManiSkill 3.0.1 tasks.

## Provenance

Based on ManiSkill's official DP baseline, `examples/baselines/diffusion_policy`
at [haosulab/ManiSkill@62ff3a5](https://github.com/haosulab/ManiSkill/tree/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/diffusion_policy)
(Apache-2.0, license in [LICENSE-ManiSkill](./LICENSE-ManiSkill)). That baseline in turn follows
Diffusion Policy by Chi et al. ([paper](https://arxiv.org/abs/2303.04137),
[code](https://github.com/real-stanford/diffusion_policy), MIT).

| File | Relation to the baseline |
| --- | --- |
| `conditional_unet1d.py` | copied unchanged |
| `policy.py` | `Agent` from `train.py`: same UNet / DDPM (100 steps, squaredcos_cap_v2, epsilon, clip) |
| `envs.py` | CPU branch of `make_env.py`, adapted to 3.0.1 |
| `data.py`, `evaluate.py`, `config.py`, `scripts/train_dp.py`, `scripts/eval_dp.py` | rewritten |

## Deliberate differences from the baseline

1. **Action normalization.** Actions are min-max scaled per dimension to [-1, 1]
   from the training demos (stats saved in the checkpoint). The baseline assumes the
   env action space already is [-1, 1] (true for `pd_ee_delta_pos`); absolute
   `pd_joint_pos` targets are not, and the DDPM sampler clips to [-1, 1].
2. **End padding for absolute control modes.** Absolute modes repeat the last
   action; the baseline only defines padding for delta modes.
3. **Final window included.** The window whose current step is the last action
   is also a training sample (the baseline's range stops one short).
4. **Sampling with replacement.** Batches are drawn uniformly with replacement from
   precomputed windows. The baseline's epoch sampler with `drop_last=True` yields no
   batches when there are fewer windows than `batch_size` (10 PickCube demos: 726 < 1024).
5. **Fixed evaluation seeds, split into validation and test.** Training-time
   evaluation (and `best.pt` selection) uses validation seeds; reported numbers come
   from `scripts/eval_dp.py` on disjoint test seeds. The baseline resets without seeds.
6. **No rendering during evaluation** unless videos are requested, so state-only
   evaluation does not need Vulkan.
7. **Logging** to JSON files under `results/<exp>/` instead of TensorBoard / W&B.

Unchanged on purpose: observations are not normalized; EMA is created as
`EMAModel(power=0.75)` exactly as in the baseline (diffusers ignores `power`
without `use_ema_warmup=True`).

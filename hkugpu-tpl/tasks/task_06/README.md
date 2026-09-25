# task_06: PlugCharger-v1

This is the working example plugin for the shared `train_template` runner. It copies the standalone task-06 model/data implementation into a plugin and adds generic engine hooks; it does not import `train_local` or `train_task_06`.

Run the shared commands from `train_template/`:

```bash
uv run python run.py prepare-data --task task_06 --conversion-workers 8
uv run python run.py train --task task_06 --model unet --seed 1
uv run python run.py evaluate --task task_06 --checkpoint runs/task_06/unet/seed_1/best.pt
uv run python run.py video --task task_06 --checkpoint runs/task_06/unet/seed_1/best.pt
uv run python run.py report --task task_06
```

The plug uses state observations and `pd_ee_delta_pose`, MLP/U-Net/Transformer noise predictors, 100-step cosine DDPM training and 12-step DDIM sampling. Demonstrations are downloaded from ManiSkill and converted to the chosen state/action representation.

# 模板使用说明

在hku gpu环境中安装uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
把当前模板git clone下来后，只需要在tasks文件夹下新建自己的task_id文件夹，然后仿照task_06结构填入自己的模型文件和相关代码即可。

以下命令都从模板目录执行。将 `<task_id>`、`<model_name>`、`<seed>` 和 `<checkpoint>` 替换为实际值；任务目录应为 `tasks/<task_id>/`。

```bash
cd 7606-train-template
```

后续的 `uv`、数据路径、权重路径都相对于这个目录。

```bash
./setup_linux_uv.sh
```

该脚本安装并检查独立环境（Python、PyTorch、ManiSkill、Diffusers、CUDA/GPU 能力）。不需要 `sudo`、`apt` 或 Conda。

```bash
uv run python run.py smoke --task <task_id>
```

执行指定任务的冒烟测试：检查任务配置中的每个模型前向、反向传播、动作采样、仿真 reset/step 和视频编码。不进行正式训练。

```bash
uv run python run.py prepare-data \
  --task <task_id> \
  --conversion-workers <conversion_workers>
```

按任务插件下载或转换专家数据。`<conversion_workers>` 是 CPU 数据转换进程数，不是 GPU 数量；没有数据准备钩子的任务不使用此命令。

```bash
uv run python run.py train \
  --task <task_id> \
  --model <model_name> \
  --seed <seed>
```

训练指定任务和模型。未显式指定的训练参数从 `tasks/<task_id>/config.py` 读取。结果位于：

```text
runs/<task_id>/<model_name>/seed_<seed>/
```

其中 `best.pt` 是验证损失最低的检查点，`latest.pt` 是最近检查点，`metrics.jsonl` 是训练日志，`run_config.json` 是本次配置。

```bash
uv run python run.py evaluate \
  --task <task_id> \
  --checkpoint runs/<task_id>/<model_name>/seed_<seed>/best.pt
```

使用指定检查点评测对应任务和模型。默认的测试 episode 数、测试 seed、并行环境数和采样步数来自该检查点保存的任务配置；CSV 和 JSON 与检查点保存在同一目录。

```bash
uv run python run.py video \
  --task <task_id> \
  --checkpoint runs/<task_id>/<model_name>/seed_<seed>/best.pt
```

生成指定检查点的策略视频，默认保存到检查点目录下的 `videos/`。

```bash
uv run python run.py report --task <task_id>
```

汇总 `runs/<task_id>/` 下各模型和训练 seed 的曲线、成功率、CSV/JSON 结果，并生成：

```text
reports/<task_id>/report.html
reports/<task_id>/training_curves.png
reports/<task_id>/model_comparison.png
```

其中 `report.html` 保留完整交互式报告，两个 `.png` 文件可以直接下载或查看。

多模型、多 seed 批量运行：

```bash
TASK=<task_id> \
MODELS="<model_name_a> <model_name_b>" \
SEEDS="<seed_a> <seed_b>" \
bash run_experiments.sh
```

也可以省略 `MODELS`，脚本会读取 `tasks/<task_id>/config.py` 的 `default_models`。所有输出目录和报告都会使用实际传入的任务、模型和 seed。

反斜杠 `\` 只表示命令换行；例如：

```bash
uv run python run.py evaluate --task <task_id> --checkpoint <checkpoint>
```

与上面的多行写法等价。


以task06为例：

```bash
cd 7606-train-template
./setup_linux_uv.sh
uv run python run.py smoke --task task_06
uv run python run.py prepare-data --task task_06 --conversion-workers 8
uv run python run.py train --task task_06 --model unet --seed 1
uv run python run.py evaluate \
  --task task_06 \
  --checkpoint runs/task_06/unet/seed_1/best.pt
uv run python run.py video \
  --task task_06 \
  --checkpoint runs/task_06/unet/seed_1/best.pt
uv run python run.py report --task task_06
uv run python run.py evaluate \
  --task task_06 \
  --checkpoint model.pt
```

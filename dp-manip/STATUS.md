# 当前状态

- 已切换为 `maniskill-demogen` RGB schema：`obs_rgb/rgb + obs_rgb/state`。
- 六任务配置已按最终任务表建立；PegInsertionSide/PlugCharger 使用 `pd_joint_pos`。
- 训练已改为集群优先：HDF5 懒加载、worker DataLoader、AMP、EMA、可恢复 checkpoint。
- Slurm 核心 96 组与条件 N=400 数组入口已建立。
- 闭环评估已改为 RGB 环境，并固定使用与数据一致的 `physx_cpu`。
- 本机没有项目的 torch/ManiSkill 环境；这里只能做静态检查。真实数据检查、GPU smoke 和
  ManiSkill rollout 必须在集群环境完成。

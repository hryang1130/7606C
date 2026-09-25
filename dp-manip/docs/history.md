# Day 1 记录（9.22）：Mac → Ubuntu → 第一条专家轨迹

> 原 `in.txt`，9.23 归档为 Markdown，内容未改动，只整理了表格格式。文中描述的是 9.22 当天的状态，其中不少事项后来已经解决：MPlib 适配已整理成补丁 `patches/mani_skill_mplib_0_2_1.patch`，正式环境已迁到 `~/Coding/dp-manip/.venv`，训练改由 wsl 承担。当前状态以 [STATUS.md](../STATUS.md) 为准。

今天我们完成了从 ManiSkill 仿真验证到成功生成第一条 PickCube 专家演示的过程。最终确定的工作方式是：Ubuntu 负责运行 MPlib 专家求解器、生成轨迹；Mac 保留为开发、查看和后续数据分析环境。但 Ubuntu 轨迹复制到 Mac 后的回放，以及 Diffusion Policy 训练，都还没有验证。

## 一、Mac 上做了什么？

在 M3 Max、36 GB 内存的 MacBook Pro 上，我们完成了基础仿真环境的搭建和验证。

| 项目       | 今天的结果                                                |
| ---------- | --------------------------------------------------------- |
| Python     | 使用 Python 3.11                                          |
| ManiSkill  | 安装并能够运行                                            |
| 图形显示   | Vulkan / MoltenVK 路径可用                                |
| 机器人仿真 | 能看到 PickCube 场景中的 Panda 机器人、红色方块和绿色目标 |
| 动作测试   | 能让机器人执行随机动作；看到的“抽搐乱动”并不是专家策略    |
| Pinocchio  | `pin` 已安装，`import pinocchio` 验证通过                 |
| MPlib      | 未能成功安装                                              |

Mac 上真正阻碍专家轨迹生成的是 MPlib 的安装依赖：安装过程被 `libclang==11.0.1` 缺少适用的 macOS ARM64 wheel 卡住。

所以我们没有继续把时间花在 Mac 的 MPlib 编译问题上，而是决定让 Linux x86-64 服务器承担专家数据生成。这是工作流分工，不是说 Mac 不能运行 ManiSkill。

## 二、Ubuntu 上做了什么？

服务器配置是 Ubuntu Server 26.04、Intel i7-8700K、GTX 1080 Ti 11 GB。

排障过程实际分成了三个阶段：

**阶段 1 · 已解决 — 让 PickCube 场景正常运行**

最初专家命令直接 Segmentation fault。我们逐步缩小范围，确认 CPU 物理系统本身可以工作，并用 `vulkaninfo` 验证 NVIDIA Vulkan 驱动可以识别 GTX 1080 Ti。随后使用 `sim_backend="physx_cpu"` 和 `render_backend="cpu"`，成功完成 PickCube 的创建、`reset()` 和关闭。

**阶段 2 · 已解决 — 找到能够加载 Panda 的 MPlib 版本**

原环境中的 mplib 0.1.1 在初始化 `ArticulatedModel` 时段错误。URDF、SRDF 文件都存在，依赖声明检查也通过；单独将 NumPy 降为 1.26.4 又造成了其他包的版本冲突，因此恢复了原环境。

我们随后创建隔离环境 `.venv-mplib-probe`，安装 mplib 0.2.1，成功加载 `panda_v2.urdf` / `panda_v2.srdf`，并初始化 Panda 规划器。

**阶段 3 · 已解决 PickCube — 适配新版 MPlib，生成专家轨迹**

ManiSkill 3.0.1 原本固定依赖 `mplib==0.1.1`，所以我们仅在测试环境覆盖了这条版本约束，并修复了 `set_base_pose()` 和 `plan_screw()` 的接口差异。

最后运行 PickCube 专家求解器，成功率显示为 1/1，生成了 H5 和 JSON 文件。

## 三、最终 Ubuntu 环境是怎么定下来的？

关键是保留了两个不同用途的虚拟环境，没有把所有尝试混在一起。

|           | 原环境 `.venv`               | 最终成功的 `.venv-mplib-probe`             |
| --------- | ---------------------------- | ------------------------------------------ |
| Python    | 3.11.15                      | 3.11.15                                    |
| ManiSkill | 3.0.1                        | 3.0.1                                      |
| MPlib     | 0.1.1                        | 0.2.1                                      |
| NumPy     | 恢复为 2.4.6                 | 由测试环境的依赖解析安装；本次未记录最终精确版本 |
| 用途      | 保留原始安装及已通过的基础仿真 | 当前可用的 PickCube 专家轨迹生成环境        |
| 状态      | 专家规划初始化会段错误       | PickCube 专家规划成功                      |

为什么最终选测试环境？因为它同时满足了两个实测条件：Panda 规划器能够初始化，而且完整 PickCube 求解流程确实生成了成功轨迹。

这里有一个重要限制：ManiSkill 3.0.1 的包元数据仍要求 MPlib 0.1.1。我们是在隔离环境中有意覆盖这条要求，再用实际运行验证兼容性；这不等于两个版本在所有 ManiSkill 任务上都已验证兼容。

### 成功环境里具体有什么？

| 类别          | 当前采用的内容                                 |
| ------------- | ---------------------------------------------- |
| 操作系统      | Ubuntu Server 26.04，Linux x86-64              |
| 处理器 / 显卡 | i7-8700K / GTX 1080 Ti 11 GB                   |
| NVIDIA 驱动   | 580.178.04                                     |
| Python 环境   | `~/Coding/dp-manip/.venv-mplib-probe`          |
| 仿真框架      | ManiSkill 3.0.1，底层 SAPIEN / PhysX           |
| 运动规划      | MPlib 0.2.1，以及安装时解析出的配套依赖        |
| 仿真后端      | `physx_cpu`                                    |
| 渲染后端      | `cpu`                                          |
| Vulkan 配置   | 执行时指定 `nvidia_icd.json`                   |
| 运行入口      | 项目目录中的 `run_cpu.py`                      |

`run_cpu.py` 是我们从 ManiSkill 官方 Panda 规划入口复制出来的本地版本，增加了 `render_backend="cpu"`。

另外，我们在测试环境的 site-packages 中做了两类 MPlib API 适配：

- 将 `set_base_pose()` 传入的 7 维数组改成 MPlib 的 `Pose` 对象。
- 将 `plan_screw()` 的目标位姿改成 `Pose`，并移除新版不接受的 `use_point_cloud` 参数。

这几处修改目前还没有整理成项目内可自动应用的补丁。如果删除并重建 `.venv-mplib-probe`，不能指望它们自动保留。

## 四、今天最终产出了什么？

**Day 1 目标达成：第一条 PickCube 专家演示**

| 指标           | 结果                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| 本次尝试成功率 | 1 / 1                                                                    |
| 轨迹长度       | 74 步                                                                    |
| 规划失败率     | 0                                                                        |
| H5 轨迹        | `demos-probe/PickCube-v1/motionplanning/pickcube_probe.h5`，29,734 字节   |
| JSON 元数据    | `demos-probe/PickCube-v1/motionplanning/pickcube_probe.json`，871 字节    |

**还没做完的事**：检查 H5 内部内容并回放轨迹；将环境修改固化为可复现配置；测试更多随机种子和其他任务；解决 GTX 1080 Ti 与当前 torch 2.14.0+cu130 的 CUDA 内核兼容问题；最后才是 Diffusion Policy 训练。

所以今天最重要的成果不是“装好了所有软件”，而是已经建立并实测了一条可运行的专家数据生成路径。下一次可以直接从检查和回放这条 74 步轨迹开始，而不必重新排查 Vulkan、URDF 和 MPlib 初始化问题。

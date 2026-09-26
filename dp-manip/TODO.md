# 下一步

1. 在集群登录节点运行 `./setup.sh`。
2. 将 `maniskill-demogen/data/dataset` 放到共享存储，运行六任务 `inspect_dataset.py`。
3. 在单张 GPU 上做 PickCube 短 smoke，记录 batch 64 的峰值显存；OOM 时全实验统一降 batch。
4. 验证 `USR1 → resume.pt → requeue` 一次。
5. 提交核心 96 组训练和固定测试评估。
6. 汇总成功率并按预注册规则判断哪些任务增加 N=400。

不要在拿到结果后更改 N 档、训练 seed 数、测试种子或 100k 步预算。

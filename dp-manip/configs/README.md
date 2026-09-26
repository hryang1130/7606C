# RGB task configs

六个 `*_rgb.toml` 是训练、checkpoint 与评估共用的权威配置。共同参数保持一致，仅任务 id、
控制模式、数据路径和回合长度不同。

`[data]` 的路径相对 `data.root`；集群上用 `--data-root` 或 `DATA_ROOT` 覆盖根目录。
`num_demos` 只允许 25/50/100/200/400，并始终取按 seed 排序后的前 N 条。

临时 smoke 可用 `--set train.total_iters=...`，但正式数据量实验必须使用配置里的 100k 步。
每个 checkpoint 保存解析后的完整配置，评估不会再次猜控制模式、相机数或 horizon。

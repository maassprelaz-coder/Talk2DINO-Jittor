# 模型权重

三个本项目训练的文本映射均为第 100 轮，大小约 3.94 MB。使用 NumPy NPZ 格式，包含线性层的权重和偏置。按 [复跑指南](REPRODUCE.md) 配合 CLIP 文本编码器和 DINOv2 骨干使用。

clone 后执行 python3 scripts/restore_assets.py，按 [分片清单](../release_assets/manifest.json)复原完整 NPZ 和 CLIP 词表。脚本逐个校验原文件 SHA-256。

| 配方 | 本仓库文件 | SHA-256 |
|---|---|---|
| 自训练 / PyTorch | `experiments/011_full_torch_seed123/projection_last.npz` | `2355c200023df9e87bb399bcfd0f0594edfeae23661e1f9ca2f6ebc28717d6e0` |
| 自训练 / Jittor | `experiments/011_full_jittor_seed123/projection_last.npz` | `92fe97eb19792ccbaf85db5e40f277e7f78a29badf5c338bcccd0ed1e0e70459` |
| Jittor 头向量均值消融 | `experiments/012_full_jittor_mean_heads_seed123/projection_last.npz` | `7097865acd496a67a4d76b00804369f8edcb5350612ac7f099baf82d243e4f49` |

原作者 mapper 的来源和校验值在 [上游版本记录](../01_证据/source_versions.json) 中。冻结骨干和作者 mapper 由 [权重转换步骤](REPRODUCE.md#2-获取固定的上游代码和权重)从原来源准备。

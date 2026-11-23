# Talk2DINO-Jittor

ICCV 2025 **Talk2DINO** 的 Jittor 复现，使用 DINOv2 ViT-B/14 with registers 和 OpenAI CLIP ViT-B/16 文本编码器。冻结图像和文本骨干，在 COCO 2014 图文对上训练文本到视觉空间的映射，并在 VOC2012 和 PASCAL Context 验证集评估开放词汇语义分割。

This repository implements the DINOv2 ViT-Base variant of Talk2DINO in Jittor. It includes reproducible evaluation scripts, three trained projection heads and numerical evidence from a matched PyTorch reference.

## 方法与项目范围

~~~mermaid
flowchart LR
    I[图像] --> D[冻结 DINOv2]
    Q[训练描述或类别名] --> C[冻结 CLIP 文本编码器]
    C --> M[文本映射 512 → 768 → 768]
    D --> V[Patch 特征和 12 个注意力头]
    V --> T[训练：配对描述先选头]
    M --> T
    T --> L[双向图文对比损失]
    V --> S[推理：Patch 与类别向量打分]
    M --> S
    S --> P[滑窗聚合与像素掩码]
~~~

迁移包含 CLIP 文本编码、DINOv2 骨干前向、头选择、InfoNCE、Adam、断点恢复、滑窗推理和分割指标。Jittor 环境独立运行模型计算。PyTorch 用于权重格式转换、复现参考和生成共享样本顺序。复现基于作者提交 [1462d0e](https://github.com/lorebianchi98/Talk2DINO/tree/1462d0ebbf3c8d2dab623a03d41548c8e6098f34)。

## 实测结果

mIoU（%）。VOC 验证集 1,449 张，Context 验证集 5,104 张。全量评测使用 FP32 和固定滑窗协议，PAMR 关闭。自训练模型均使用第 100 轮的最终映射。

| 权重与框架 | VOC20 | VOC21 | Context59 | Context60 |
|---|---:|---:|---:|---:|
| 作者权重 · PyTorch | 87.1055 | 61.4485 | 39.8018 | 35.1342 |
| 同一作者权重 · Jittor | 87.1055 | 61.4485 | 39.8018 | 35.1342 |
| 自训练主模型 · PyTorch | 87.4271 | 58.7360 | 39.2733 | 34.2089 |
| 自训练主模型 · Jittor | 87.4000 | 58.6522 | 39.2582 | 34.2192 |
| 自训练均值头消融 · Jittor | 89.1771 | 57.4327 | 36.9554 | 32.1491 |

- **框架对照。** 相同作者映射权重时，四个指标在表中显示精度一致。同配方各自训练后，四口径最大差为 0.0839 个百分点。
- **机制消融。** 最大匹配头优于均值头的指标有 VOC21、Context59 和 Context60，VOC20 则由均值头取得更高结果。固定自训练 Jittor 权重，在 VOC21 上开启背景清理使 mIoU 从 56.2365% 升至 58.6522%。
- **实验范围。** 自训练 Jittor 的 VOC21 比作者权重低约 2.80 个百分点，具体原因尚未确定。主实验采用种子 123、两套验证集和不含 PAMR 的条件。同机独立 Python 环境复跑了 VOC 前 5 张，详细范围见[复跑收据](experiments/independent_venv_check/receipt.json)。

[最终实验审计](experiments/final_experiment_audit.json)记录 177 项历史检查和主表数据。另有[结构化表格](experiments/最终实验结果.csv)及按实验区分的训练和评测原始指标。三组 mapper 各训练 100 轮、323,600 次更新，COCO 训练集为 82,783 张图片、414,113 条图文描述。训练日志重现了配方与每轮指标，完整数据和预测文件可按复跑文档重新生成。

## 获取代码与复跑

~~~bash
git clone https://github.com/maassprelaz-coder/Talk2DINO-Jittor.git
cd Talk2DINO-Jittor
~~~

- **检查已保存结果。** 无数据和 GPU 时可阅读上表、[最终实验审计](experiments/final_experiment_audit.json)及各运行目录的 metrics.json。可运行 python -m unittest discover -s tests -v 检查混淆矩阵与 IoU 行为，需要 NumPy。
- **推理和完整实验。** [复跑指南](docs/REPRODUCE.md)列出了已验证环境、上游提交、训练/评估数据来源、权重准备、VOC 五图推理、全量评测和三组训练命令。
- **下载 mapper。** 采用 [权重索引](docs/WEIGHTS.md)中的本项目已训练 NPZ，或在 PyTorch 参考环境转换原作者发布权重。mapper 使用两套单独的冻结骨干。

| 目录 | 内容 |
|---|---|
| src/talk2dino_jittor | Jittor 模型、预处理、推理、分词与评价 |
| scripts | 上游权重转换、数据准备、特征缓存、训练、评价与对齐 |
| environment | 已实测依赖和环境版本 |
| experiments | 各运行的指标、协议、权重 SHA-256 和三个自训练 mapper |
| 01_证据 | 固定版本、兼容补丁和数据来源索引 |
| tests | 像素指标的单元测试 |

## 引用与许可

论文由 Luca Barsellotti、Lorenzo Bianchi、Nicola Messina、Fabio Carrara、Marcella Cornia、Lorenzo Baraldi、Fabrizio Falchi、Rita Cucchiara 提出。引用原方法请使用 [CVF 会议论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Barsellotti_Talking_to_DINO_Bridging_Self-Supervised_Vision_Backbones_with_Language_for_ICCV_2025_paper.html)及其 BibTeX。上游代码与模型由 [Talk2DINO 作者仓库](https://github.com/lorebianchi98/Talk2DINO)、[DINOv2](https://github.com/facebookresearch/dinov2)、[OpenAI CLIP](https://github.com/openai/CLIP)提供。

项目实现许可见 [LICENSE](LICENSE)。随包的 CLIP 分词器、词表及参考工具保留各自的[第三方许可](third_party/licenses/)。完整数据集和上游原始权重从其来源获取，使用时遵循对应条款。

本仓库为人工智能实践课大作业二复现交付。测试硬件为 RTX 5080 Laptop 16 GB，本机 WSL2 Ubuntu 24.04。环境与运行记录见[复跑指南](docs/REPRODUCE.md)，维护与复现问题可在本仓库 Issues 中记录。

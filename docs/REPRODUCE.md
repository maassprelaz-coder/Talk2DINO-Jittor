# Talk2DINO-Jittor 复跑指南

下面从仓库根目录执行。默认使用 Linux 或 WSL2、NVIDIA GPU、CUDA 工具链及 Python 3.10。本轮完成的是同机独立环境的小样本验收，更多系统和 GPU 组合需要重新检查兼容性。

## 1. 还原仓库资产和建立两个独立环境

在仓库根目录用 Python 3 还原三个 mapper 和 CLIP 词表。脚本拼接随仓库下载的文件，并核对每个原始 SHA-256。

~~~bash
python3 scripts/restore_assets.py
~~~

记录环境为 Ubuntu 24.04 / WSL2，RTX 5080 Laptop 16 GB，Python 3.10.22，CUDA 编译器 12.8.93，Jittor 1.3.11.0，Torch 2.8.0+cu128。请在创建环境前确保运行 nvidia-smi 和 nvcc --version 成功。以下用 python3.10 代表已安装的合适解释器。

~~~bash
python3.10 -m venv .venv-jittor
python3.10 -m venv .venv-torch
JITTOR_PY="$PWD/.venv-jittor/bin/python"
TORCH_PY="$PWD/.venv-torch/bin/python"

"$JITTOR_PY" -m pip install -r environment/jittor_snapshot.txt
"$TORCH_PY" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$TORCH_PY" -m pip install numpy==1.26.4 pillow pyyaml requests tqdm ftfy regex opencv-python-headless==4.11.0.86
~~~

使用与显卡相容的 NVIDIA 驱动及 CUDA。需要时设置 nvcc_path 为 nvcc 的实际绝对路径，Linux 下确保相容运行库对当前环境可见。环境出现课程实验中记录的 CUDA 12.8 / sm_120 和 NPP 问题时，可在 Jittor 虚拟环境内应用局部补丁。补丁检查具体源码结构，保留原始文件。变更摘要见 [记录](../01_证据/)。

~~~bash
"$JITTOR_PY" scripts/patch_jittor_cuda128.py
"$JITTOR_PY" scripts/patch_jittor_headers.py
"$JITTOR_PY" -c 'import jittor as jt; jt.flags.use_cuda=1; print(jt.array([1,2,3]).sum().item())'
"$TORCH_PY" -c 'import torch; print(torch.cuda.is_available())'
~~~

## 2. 获取固定的上游代码和权重

每个 git clone 在目标不存在时运行一次。固定提交列表也见 [source_versions.json](../01_证据/source_versions.json)。CLIP 需从 OpenAI 仓库安装。

~~~bash
mkdir -p upstream checkpoints
git clone https://github.com/lorebianchi98/Talk2DINO.git upstream/Talk2DINO
git -C upstream/Talk2DINO checkout 1462d0ebbf3c8d2dab623a03d41548c8e6098f34
git clone https://github.com/facebookresearch/dinov2.git upstream/dinov2
git -C upstream/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
git clone https://github.com/openai/CLIP.git upstream/CLIP
git -C upstream/CLIP checkout d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
"$TORCH_PY" -m pip install ./upstream/CLIP
export TALK2DINO_RUNTIME="$PWD/.runtime"
"$TORCH_PY" scripts/export_backbone_reference.py --root "$PWD"
~~~

转换器由固定上游代码获取 DINOv2 ViT-B/14 with registers 与 CLIP ViT-B/16，将 DINOv2 和 CLIP 文本端分别导出为 checkpoints/dinov2_vitb14_reg_fp32.npz、checkpoints/clip_vitb16_text_fp32.npz。记录见 [历史骨干转换参考](../experiments/001_backbone_alignment/reference_448.json)。

作者映射位于固定上游仓库的 weights/vitb_mlp_infonce.pth。转换前核对 SHA-256，若仓库提供 LFS 指针，先用该仓库指明的方式取得实际模型文件。完成后生成作者映射和供对齐检查的合成样本。

~~~bash
sha256sum upstream/Talk2DINO/weights/vitb_mlp_infonce.pth
"$TORCH_PY" scripts/export_projection_reference.py --root "$PWD"
"$TORCH_PY" scripts/export_initial_projection.py --root "$PWD" --seed 123
"$JITTOR_PY" scripts/check_projection_alignment.py --root "$PWD" --device cuda
"$JITTOR_PY" scripts/check_backbone_alignment.py --root "$PWD" --model both --size 448
~~~

对应作者 PTH SHA-256 是 5f6efcc520f22916ecf3dfdfb01d6f57e701c1c2e1feda71a80d89e3cd752f0f。只做自训练 Jittor 权重推理时，可以跳过作者 mapper 和训练初始化的转换，使用仓库中指定的 mapper。骨干转换仍需执行。

## 3. VOC 和 Context 数据与评测

先准备 VOC，转换程序会校验官方 MD5，并输出所需验证集图像、掩码与名称。数据源是公开镜像中的原始 VOC2012 包。五图运行使用新的 run-name，输出在 experiments/demo_voc5/，其中 full_benchmark 为 false。完整评测移除 --limit 并使用新的 run-name。

~~~bash
"$TORCH_PY" scripts/prepare_voc2012.py --root "$PWD"

JITTOR_MAPPER="$PWD/experiments/011_full_jittor_seed123/projection_last.npz"
"$JITTOR_PY" scripts/evaluate_voc.py --root "$PWD" --framework jittor \
    --weights "$JITTOR_MAPPER" --run-name demo_voc5 --limit 5

"$JITTOR_PY" scripts/evaluate_voc.py --root "$PWD" --framework jittor \
    --weights "$JITTOR_MAPPER" --run-name reproduce_jittor_final_voc
~~~

Context 使用 PyTorch-Encoding 已准备的验证掩码数据，校验发布 SHA-1 后，与独立来源的 5,104 张验证列表比对。图像从同一份通过官方 MD5 检查的 VOC 压缩包提取。背景阈值 0.55，关闭背景清理和 PAMR。

~~~bash
"$TORCH_PY" scripts/prepare_context_val.py --root "$PWD"
"$JITTOR_PY" scripts/evaluate_context.py --root "$PWD" --framework jittor \
    --weights "$JITTOR_MAPPER" --run-name reproduce_jittor_final_context
~~~

评测作者 mapper 时省略 --weights，并使用新 run-name。复跑 PyTorch 时使用 Torch 解释器和 --framework torch。VOC20、VOC21（包含开/关清理）和 Context59/60 的口径见 [协议实现](../src/talk2dino_jittor/benchmark_protocol.py)和各 metrics.json。训练和评测中不得用 VOC 或 Context 标注挑训练轮次、阈值或模型。这个项目按固定配置取第 100 轮。

## 4. 完整 COCO 训练

COCO images 和 captions 来源为 [COCO 2014](https://cocodataset.org/#download)，本项目保留解压步骤和校验代码。当前脚本启动全量下载时检查可用空间至少 75 GiB，建议准备更多空间放训练缓存。先取得 annotations_trainval2014.zip 内的两个 captions 文件，再运行校验解压流程。

~~~bash
mkdir -p data/coco2014
curl -fL https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip \
    -o data/coco2014/annotations_trainval2014.zip
unzip -n data/coco2014/annotations_trainval2014.zip \
    annotations/captions_train2014.json annotations/captions_val2014.json -d data/coco2014
"$TORCH_PY" scripts/prepare_full_coco.py --root "$PWD"
"$TORCH_PY" scripts/export_training_orders.py --root "$PWD" --seed 123
"$JITTOR_PY" scripts/extract_full_features.py --root "$PWD" --framework jittor --image-batch 8
"$TORCH_PY" scripts/extract_full_features.py --root "$PWD" --framework torch --image-batch 8
"$JITTOR_PY" scripts/check_full_feature_alignment.py
~~~

使用缓存特征训练映射。三个运行的输入和设置应保持一致；mean_heads 仅改变训练阶段的聚合方式。记录的 100 轮中每轮 3,236 步。

~~~bash
"$JITTOR_PY" scripts/train_full_mapper.py --root "$PWD" --framework jittor \
    --features "$PWD/features/full_jittor" --run-name reproduce_train_jittor --epochs 100 --seed 123
"$TORCH_PY" scripts/train_full_mapper.py --root "$PWD" --framework torch \
    --features "$PWD/features/full_torch" --run-name reproduce_train_torch --epochs 100 --seed 123
"$JITTOR_PY" scripts/train_full_mapper.py --root "$PWD" --framework jittor \
    --features "$PWD/features/full_jittor" --run-name reproduce_train_mean_heads --epochs 100 --seed 123 \
    --alignment mean_heads
~~~

对应的新运行目录里，projection_last.npz 是最终映射。续训使用相同参数、相同 run-name 和 --resume 指向对应 latest.npz。按第 3 节评测三个新 mapper。固定作者配置 save_best_model:false 是结论使用第 100 轮的依据。

## 5. 保存的证据

[最终审计](../experiments/final_experiment_audit.json)及 [结果表](../experiments/最终实验结果.csv)记录原工作区的全量评测。evaluate_voc.py 和 evaluate_context.py 保存每口径逐类 IoU、混淆矩阵与真实预测。原 177 项审计包括所有预测图和真值像素计数，它的重跑需要重新生成对应数据及目录，归档 JSON 可离线阅读。复跑输出用新目录便于对照。

平台和版本信息来自 [环境快照](../environment/environment.json)。同机独立 Jittor venv 的 5 图验证及数值范围见 [复跑收据](../experiments/independent_venv_check/receipt.json)。骨干、完整训练数据、全量特征和预测掩码按来源准备；本仓库只保存小体积、可核对的项目产物。

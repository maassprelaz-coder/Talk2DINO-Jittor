"""Audit completed runs against actual label histograms, then emit final result tables."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from range_download import atomic_json,sha256
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"experiments";checks=[]
def require(name,condition):
    checks.append({"name":name,"pass":bool(condition)})
    if not condition:raise RuntimeError("Failed experiment audit: "+name)
def load(path):return json.loads(Path(path).read_text())
def histograms():
    result={}
    for dataset,year,list_path,mask_folder,classes in [
        ("voc","VOC2012","ImageSets/Segmentation/val.txt","SegmentationClass",21),
        ("context","VOC2010","ImageSets/SegmentationContext/val.txt","SegmentationClassContext",60)]:
        data=ROOT/"data/VOCdevkit"/year;names=(data/list_path).read_text().splitlines()
        require(dataset+"_unique_full_split",len(names)==len(set(names))==(1449 if dataset=="voc" else 5104))
        counts=np.zeros(classes,np.int64)
        for name in names:
            with Image.open(data/mask_folder/(name+".png")) as image:mask=np.asarray(image)
            valid=mask[mask!=255]
            require_name=False
            if np.any(valid>=classes):raise RuntimeError("Unexpected ground-truth label")
            counts+=np.bincount(valid,minlength=classes)
        result[dataset]={"names":set(names),"hist":counts}
    return result
truth=histograms()
runs=[
 ("official_torch","官方权重 / PyTorch",None,"006_voc_official_torch","010_context_official_torch"),
 ("official_jittor","官方权重 / Jittor",None,"006_voc_official_jittor","010_context_official_jittor"),
 ("full_torch","自训练 / PyTorch","011_full_torch_seed123","011_full_torch_seed123_final_voc","011_full_torch_seed123_final_context"),
 ("full_jittor","自训练 / Jittor","011_full_jittor_seed123","011_full_jittor_seed123_final_voc","011_full_jittor_seed123_final_context"),
 ("mean_heads","Jittor 头向量均值消融","012_full_jittor_mean_heads_seed123","012_full_jittor_mean_heads_seed123_final_voc","012_full_jittor_mean_heads_seed123_final_context")]
rows=[];train_configs={}
official_mapper=sha256(BASE/"000_projection_alignment/fixtures/weights.npz")
for key,label,train_name,voc_name,context_name in runs:
    row={"key":key,"method":label}
    mapper=official_mapper
    if train_name:
        train=load(BASE/train_name/"metrics.json");selection=load(BASE/train_name/"primary_selection.json")
        require(key+"_full_100_epochs",train["complete"] and train["full_training"] and train["state"]["epoch"]==100)
        require(key+"_full_sample_counts",(train["train_images"],train["train_captions"],train["validation_captions"])==(82783,414113,202654))
        require(key+"_exact_steps",train["state"]["step"]==323600)
        mapper=sha256(BASE/train_name/"projection_last.npz")
        require(key+"_primary_final_weight",selection["epochs"]==100 and selection["primary_mapper"]=="projection_last.npz"
                and selection["primary_mapper_sha256"]==mapper and selection["verified_equal_to_latest_checkpoint"])
        config=train["config"];train_configs[key]=config
        require(key+"_fixed_recipe",config["epochs"]==100 and config["batch_size"]==128 and config["lr"]==1e-4
                and config["temperature"]==.07 and config["scope"]=="full")
        row["train_seconds"]=train["state"]["seconds"];row["train_run"]=train_name
    for dataset,name in [("voc",voc_name),("context",context_name)]:
        result=load(BASE/name/"metrics.json");config=result["config"];expected_count=1449 if dataset=="voc" else 5104
        require(key+"_"+dataset+"_complete_full",result["complete"] and result["full_benchmark"] and result["images"]==expected_count)
        require(key+"_"+dataset+"_correct_weight",config["mapper_sha256"]==mapper)
        require(key+"_"+dataset+"_fp32_no_pamr",config["precision"]=="FP32" and not config["pamr"])
        require(key+"_"+dataset+"_threshold",config["background_threshold"]==.55)
        if dataset=="voc":
            require(key+"_voc_bg_clean_policy",not config["protocols"]["voc20"]["clean"]
                    and config["protocols"]["voc21"]["clean"] and not config["protocols"]["voc21_no_clean"]["clean"])
            expected={"voc20":truth["voc"]["hist"][1:],"voc21":truth["voc"]["hist"],
                      "voc21_no_clean":truth["voc"]["hist"]}
        else:
            require(key+"_context_clean_disabled",not config["with_bg_clean"])
            expected={"context59":truth["context"]["hist"][1:],"context60":truth["context"]["hist"]}
        for protocol,hist in expected.items():
            metric=result["results"][protocol];matrix=np.asarray(metric["confusion"],np.int64)
            require(key+"_"+protocol+"_matrix_shape",matrix.shape==(len(hist),len(hist)) and np.all(matrix>=0))
            require(key+"_"+protocol+"_all_gt_pixels",np.array_equal(matrix.sum(1),hist))
            tp=np.diag(matrix).astype(np.float64);union=matrix.sum(0)+matrix.sum(1)-tp
            iou=np.divide(tp,union,out=np.full(len(tp),np.nan),where=union!=0)
            require(key+"_"+protocol+"_miou_recomputed",abs(float(np.nanmean(iou))-metric["miou"])<1e-12)
            predictions={p.stem for p in (BASE/name/"predictions"/protocol).glob("*.png")}
            require(key+"_"+protocol+"_all_predictions",predictions==truth[dataset]["names"])
            row[protocol]=float(metric["miou"]*100)
        row[dataset+"_metrics"]=name+"/metrics.json"
    rows.append(row)
by_key={row["key"]:row for row in rows};protocols=["voc20","voc21","context59","context60"]
for setting in ["seed","epochs","batch_size","lr","temperature","loss","order_sha256","initial_weight_sha256","alignment"]:
    require("paired_training_"+setting,train_configs["full_torch"][setting]==train_configs["full_jittor"][setting])
differences={p:by_key["full_jittor"][p]-by_key["full_torch"][p] for p in protocols}
require("paired_training_miou_within_one_point",max(abs(x) for x in differences.values())<=1.)
clean_gain=by_key["full_jittor"]["voc21"]-by_key["full_jittor"]["voc21_no_clean"]
head_gain={p:by_key["full_jittor"][p]-by_key["mean_heads"][p] for p in protocols}
audit={"pass":True,"checks":checks,"check_count":len(checks),"rows":rows,
       "primary_selection":"epoch 100; official save_best_model=false",
       "paired_miou_delta_percentage_points":differences,
       "max_paired_delta_percentage_points":max(abs(x) for x in differences.values()),
       "background_clean_gain_voc21_points":clean_gain,"max_head_minus_mean_head_points":head_gain,
       "ground_truth_pixel_histograms":{d:x["hist"].tolist() for d,x in truth.items()},
       "cloud_spent_cny":0,"scope":"Completed experiment audit; report/PPT/clean-machine delivery is separate"}
atomic_json(BASE/"final_experiment_audit.json",audit)
with (BASE/"最终实验结果.csv").open("w",encoding="utf-8-sig",newline="") as stream:
    writer=csv.writer(stream);writer.writerow(["方法","VOC20 mIoU (%)","VOC21 mIoU (%)","Context59 mIoU (%)","Context60 mIoU (%)"])
    for row in rows:writer.writerow([row["method"]]+[f"{row[p]:.6f}" for p in protocols])
lines=["# 最终实验结果与验收","",
"更新：2026-10-25。排队的主训练、两项消融和全部评测均已完成。本文汇总实验结果，课程最终报告、PPT 和讲解学习包仍需继续整理。","",
"## 主结果","",
"全部使用 FP32、固定官方模板和滑窗协议，不使用 PAMR。VOC 验证集 1,449 张，Context 验证集 5,104 张。自训练与头均值消融均完成完整 COCO 的 100 轮，主结果使用第 100 轮权重。","",
"| 模型与框架 | VOC20 | VOC21 | Context59 | Context60 |",
"|---|---:|---:|---:|---:|"]
for row in rows:lines.append("| "+row["method"]+" | "+" | ".join(f"{row[p]:.4f}" for p in protocols)+" |")
lines+=["","以上均为 mIoU（%）。官方权重结果与自训练结果分列，未将作者训练好的参数当作本项目的训练成绩。","",
"## 两框架的自训练对照","",
f"相同初始化、样本顺序和主配置下，四个口径的最大 mIoU 差为 **{audit['max_paired_delta_percentage_points']:.4f} 个百分点**。这满足计划中自训练对照差不超过约 1 个百分点的验收目标，但不代表两边参数或像素输出逐位相同。","",
"## 两项消融","",
f"- 背景清理：固定 Jittor 自训练权重，VOC21 从 {by_key['full_jittor']['voc21_no_clean']:.4f}% 升至 {by_key['full_jittor']['voc21']:.4f}%，提升 **{clean_gain:.4f} 个百分点**。",
"- 头选择：将训练中的最大匹配头选择替换为 12 头视觉向量均值，其他设置和样本顺序相同。这是本项目明确定义的消融，不冒充论文的平均注意力路径。","",
"| 指标 | 最大匹配头 − 头向量均值（百分点） |",
"|---|---:|"]
for p in protocols:lines.append(f"| {p} | {head_gain[p]:+.4f} |")
lines+=["",
"头向量均值在 VOC20 更高，最大匹配头在 VOC21、Context59/60 更高。不能写成某种头选择在所有数据集上都更好。","",
"## 结果边界与诚实报告","",
f"- 自训练 Jittor 的 VOC21 比官方权重低 {by_key['official_jittor']['voc21']-by_key['full_jittor']['voc21']:.4f} 个百分点。该差距保留，不通过挑选分割验证集上的最佳轮或阈值来消除。",
"- 原因尚未确定。两框架自训练非常接近，有助于缩小排查范围，但不足以证明差距必然来自某一种精度或环境差异。",
"- 主训练种子为 123，没有把一次运行写成多种子均值；更换初始化或额外训练不属于这轮已完成的主实验。",
"- 最初误评过 COCO 验证损失最低的第 97 轮。已纠正为官方 save_best_model:false 对应的第 100 轮，并单独保留旧结果作补充，详见 [修正记录](checkpoint_selection_correction.json)。",
"- 训练和评估日志的耗时包含缓存读取、保存和温控间歇，不用于声称严格的框架速度优势。",
"- 单机环境和运行器仍存在机器相关设置；独立干净环境复跑验收、最终报告、PPT 和课程提交不是此次队列完成状态的一部分。","",
"## 已完成的验收","",
f"共通过 {len(checks)} 项文件及数值检查，包括完整训练数量、100 轮与步数、最终权重哈希、评价图片数量、全部预测文件、真实标注的每类像素总数以及从混淆矩阵重新计算 mIoU。全量特征检查也保留原 atol=1e-5、rtol=1e-4，未因结果扩大阈值。","",
"训练数据：82,783 张图片、414,113 条描述；每组训练 323,600 次参数更新。验证损失监测使用 202,654 条描述。Jittor、PyTorch 主模型及均值头消融都按该完整规模运行。","",
"## 文件入口","",
"- [完整验收记录](final_experiment_audit.json)",
"- [结果表 CSV](最终实验结果.csv)",
"- [全量特征数值对照](004_full_data/full_feature_alignment.json)",
"- [Jittor 最终权重核验](011_full_jittor_seed123/primary_selection.json)",
"- [PyTorch 最终权重核验](011_full_torch_seed123/primary_selection.json)",""]
for row in rows:
    lines.append(f"- {row['method']}：[VOC 原始结果]({row['voc_metrics']}) · [Context 原始结果]({row['context_metrics']})")
lines+=["","当前全部实验使用本机 GPU，云 GPU / 云存储支出为 0 元。队列和温控守护已结束，电脑不需要为这轮实验继续保持开机。",""]
(BASE/"最终实验结果.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps({"pass":True,"check_count":len(checks),"main_results":by_key["full_jittor"],
                  "max_framework_difference":audit["max_paired_delta_percentage_points"],
                  "background_gain":clean_gain},indent=2),flush=True)

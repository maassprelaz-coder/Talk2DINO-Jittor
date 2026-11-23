"""Evaluate official or self-trained mapper on the complete official VOC val split."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from talk2dino_jittor.benchmark_protocol import VOC_CLASSES,TEMPLATES,PROTOCOLS,resize_outer,windows,normalize_crop,voc_target
from talk2dino_jittor.metrics import ConfusionMatrix
from evaluation_backends import JittorBackend,TorchBackend
from range_download import atomic_json,sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--framework",choices=["jittor","torch"],required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--run-name",required=True)
    p.add_argument("--limit",type=int,default=0,help="Debug only; partial results never marked full benchmark")
    p.add_argument("--save-scores",type=int,default=0)
    a=p.parse_args();root=a.root.resolve()
    data=root/"data/VOCdevkit/VOC2012"
    manifest=json.loads((data/"manifest.json").read_text())
    names=(data/"ImageSets/Segmentation/val.txt").read_text().splitlines()
    if len(names)!=1449 or manifest["images"]!=1449:raise RuntimeError("Unexpected official validation split")
    weights=a.weights or root/"experiments/000_projection_alignment/fixtures/weights.npz"
    out=root/"experiments"/a.run_name;out.mkdir(parents=True,exist_ok=True)
    config={"dataset":"VOC2012","split":"segmentation val","full_images":1449,"evaluated_images":a.limit or 1449,
            "framework":a.framework,"mapper_sha256":sha256(weights),"precision":"FP32","pamr":False,
            "outer_resize":{"max_long":2048,"short":448,"keep_ratio":True,"interpolation":"OpenCV bilinear uint8"},
            "crop":448,"stride":224,"within_crop_align_corners":True,"final_align_corners":False,
            "templates":list(TEMPLATES),"class_names":list(VOC_CLASSES),"background_threshold":.55,
            "protocols":PROTOCOLS,"void":255,"dataset_archive_md5":manifest["official_archive_md5"]}
    names=names[:a.limit] if a.limit else names
    state_path=out/"progress.json"
    metrics={key:ConfusionMatrix(protocol["classes"]) for key,protocol in PROTOCOLS.items()}
    cursor=0;seconds=0.
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state["config"]!=config:raise RuntimeError("Evaluation configuration changed; use a new run name")
        cursor=state["done"];seconds=state["seconds"]
        for key in metrics:metrics[key].matrix=np.array(state["confusion"][key],dtype=np.int64)
    backend=(JittorBackend if a.framework=="jittor" else TorchBackend)(
        root,weights,[template.format(name) for name in VOC_CLASSES for template in TEMPLATES],len(TEMPLATES))
    started=time.monotonic()
    def checkpoint():
        atomic_json(state_path,{"config":config,"done":cursor,"seconds":seconds+time.monotonic()-started,
                               "confusion":{k:v.matrix.tolist() for k,v in metrics.items()}})
    for index in range(cursor,len(names)):
        name=names[index];bgr=cv2.imread(str(data/"JPEGImages"/(name+".jpg")),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError("Unreadable VOC image "+name)
        original_shape=bgr.shape[:2];rgb=resize_outer(bgr)[:,:,::-1].copy()
        h,w=rgb.shape[:2]
        scores={key:np.zeros((protocol["classes"],h,w),np.float32) for key,protocol in PROTOCOLS.items()}
        counts=np.zeros((h,w),np.float32)
        for y1,y2,x1,x2 in windows(h,w):
            crop=rgb[y1:y2,x1:x2]
            masks=backend.crop_scores(normalize_crop(crop),crop.shape[:2])
            for key,protocol in PROTOCOLS.items():
                mask=masks[protocol["clean"]]
                if protocol["background"]:
                    mask=np.concatenate([np.full((1,*crop.shape[:2]),.55,np.float32),mask],axis=0)
                scores[key][:,y1:y2,x1:x2]+=mask
            counts[y1:y2,x1:x2]+=1
        if np.any(counts==0):raise RuntimeError("Uncovered sliding-window pixels")
        target=np.asarray(Image.open(data/"SegmentationClass"/(name+".png")))
        for key,protocol in PROTOCOLS.items():
            restored=backend.restore(scores[key]/counts[None],original_shape)
            prediction=restored.argmax(0).astype(np.uint8)
            metrics[key].update(prediction,voc_target(target,protocol["background"]))
            pred_dir=out/"predictions"/key;pred_dir.mkdir(parents=True,exist_ok=True)
            Image.fromarray(prediction).save(pred_dir/(name+".png"))
            if index<a.save_scores:
                score_dir=out/"fixtures";score_dir.mkdir(exist_ok=True)
                np.save(score_dir/(name+"_"+key+".npy"),restored)
        cursor=index+1
        if cursor==1 or cursor%25==0 or cursor==len(names):
            checkpoint()
            print(json.dumps({"done":cursor,"total":len(names),
                  "miou_running":{k:round(v.compute()["miou"]*100,4) for k,v in metrics.items()},
                  "seconds":seconds+time.monotonic()-started}),flush=True)
    result={"config":config,"complete":cursor==len(names),"full_benchmark":not bool(a.limit),
            "images":cursor,"seconds":seconds+time.monotonic()-started,
            "results":{k:v.compute() for k,v in metrics.items()},
            "weight_role":"official pretrained mapper" if a.weights is None else "user-specified mapper (see hash)",
            "caution":"FP32 protocol port; partial debug scores are not a full benchmark"}
    atomic_json(out/"metrics.json",result)
    print(json.dumps({k:v["miou"]*100 for k,v in result["results"].items()},indent=2),flush=True)
if __name__=="__main__":main()

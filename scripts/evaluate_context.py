"""Context59/60 validation using the pinned no-PAMR, no-cleaning protocol."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from talk2dino_jittor.benchmark_protocol import TEMPLATES,resize_outer,windows,normalize_crop,voc_target
from talk2dino_jittor.metrics import ConfusionMatrix
from evaluation_backends import JittorBackend,TorchBackend
from range_download import atomic_json,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--framework",choices=["jittor","torch"],required=True)
    p.add_argument("--weights",type=Path);p.add_argument("--run-name",required=True)
    p.add_argument("--limit",type=int,default=0);p.add_argument("--save-scores",type=int,default=0)
    a=p.parse_args();root=a.root.resolve();data=root/"data/VOCdevkit/VOC2010"
    manifest=json.loads((data/"context_manifest.json").read_text())
    names=(data/"ImageSets/SegmentationContext/val.txt").read_text().splitlines()
    if len(names)!=5104 or not manifest["independent_5104_validation_ids_equal"]:
        raise RuntimeError("Context validation split not verified")
    classes=manifest["classes59"]
    weights=a.weights or root/"experiments/000_projection_alignment/fixtures/weights.npz"
    config={"dataset":"PASCAL Context","split":"validation","full_images":5104,"evaluated_images":a.limit or 5104,
            "framework":a.framework,"mapper_sha256":sha256(weights),"precision":"FP32","pamr":False,
            "with_bg_clean":False,"outer_resize":[2048,448],"crop":448,"stride":224,
            "within_crop_align_corners":True,"final_align_corners":False,"background_threshold":.55,
            "templates":list(TEMPLATES),"class_names59":classes,
            "labels":"Context59 ignores raw 0; Context60 includes raw 0; raw labels are 0..59",
            "dataset_archive_sha1":manifest["published_archive_sha1"]}
    out=root/"experiments"/a.run_name;out.mkdir(parents=True,exist_ok=True)
    metrics={key:ConfusionMatrix(n) for key,n in [("context59",59),("context60",60)]}
    cursor=0;previous=0.
    progress=out/"progress.json"
    if progress.exists():
        state=json.loads(progress.read_text())
        if state["config"]!=config:raise RuntimeError("Context evaluation configuration changed")
        cursor=state["done"];previous=state["seconds"]
        for key in metrics:metrics[key].matrix=np.array(state["confusion"][key],np.int64)
    backend=(JittorBackend if a.framework=="jittor" else TorchBackend)(
        root,weights,[template.format(name) for name in classes for template in TEMPLATES],len(TEMPLATES))
    names=names[:a.limit] if a.limit else names;started=time.monotonic()
    for index in range(cursor,len(names)):
        name=names[index];bgr=cv2.imread(str(data/"JPEGImages"/(name+".jpg")),cv2.IMREAD_COLOR)
        if bgr is None:raise RuntimeError("Unreadable Context image")
        original_shape=bgr.shape[:2];rgb=resize_outer(bgr)[:,:,::-1].copy();h,w=rgb.shape[:2]
        scores=np.zeros((60,h,w),np.float32);count=np.zeros((h,w),np.float32)
        for y1,y2,x1,x2 in windows(h,w):
            crop=rgb[y1:y2,x1:x2]
            masks=backend.crop_scores(normalize_crop(crop),crop.shape[:2],with_clean=False)[False]
            scores[:,y1:y2,x1:x2]+=np.concatenate([np.full((1,*crop.shape[:2]),.55,np.float32),masks])
            count[y1:y2,x1:x2]+=1
        if np.any(count==0):raise RuntimeError("Uncovered Context pixel")
        restored=backend.restore(scores/count[None],original_shape)
        target=np.asarray(Image.open(data/"SegmentationClassContext"/(name+".png")))
        for key,values,bg in [("context59",restored[1:],False),("context60",restored,True)]:
            prediction=values.argmax(0).astype(np.uint8)
            metrics[key].update(prediction,voc_target(target,bg))
            folder=out/"predictions"/key;folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(prediction).save(folder/(name+".png"))
            if index<a.save_scores:
                fixtures=out/"fixtures";fixtures.mkdir(exist_ok=True)
                np.save(fixtures/(name+"_"+key+".npy"),values)
        cursor=index+1
        if cursor==1 or cursor%25==0 or cursor==len(names):
            atomic_json(progress,{"config":config,"done":cursor,"seconds":previous+time.monotonic()-started,
                                  "confusion":{k:v.matrix.tolist() for k,v in metrics.items()}})
            print(json.dumps({"done":cursor,"total":len(names),"seconds":previous+time.monotonic()-started,
                  "miou_running":{k:v.compute()["miou"]*100 for k,v in metrics.items()}}),flush=True)
    atomic_json(out/"metrics.json",{"config":config,"complete":True,"full_benchmark":not bool(a.limit),
        "images":cursor,"seconds":previous+time.monotonic()-started,"results":{k:v.compute() for k,v in metrics.items()},
        "weight_role":"official pretrained mapper" if a.weights is None else "user-specified mapper (see hash)"})
    print("Context evaluation complete",flush=True)
if __name__=="__main__":main()

"""Formal all-caption 100-epoch training with provenance and portable resume."""
import argparse,json,math,signal,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from range_download import atomic_json,sha256
from training_backends import JittorTrainer,TorchTrainer

def save_npz(path,arrays):
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:np.savez(stream,**arrays)
    temporary.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--framework",choices=["jittor","torch"],required=True)
    p.add_argument("--features",type=Path,required=True);p.add_argument("--run-name",required=True)
    p.add_argument("--epochs",type=int,default=100);p.add_argument("--seed",type=int,default=123)
    p.add_argument("--alignment",choices=["max_score","mean_heads"],default="max_score")
    p.add_argument("--probe",action="store_true");p.add_argument("--max-steps",type=int,default=0)
    p.add_argument("--resume",type=Path);p.add_argument("--checkpoint-every",type=int,default=5000)
    a=p.parse_args();root=a.root.resolve();features=a.features.resolve()
    if a.epochs!=100 and not a.probe:raise RuntimeError("Formal recipe is 100 epochs; use --probe for reduced tests")
    if a.max_steps and not a.probe:raise RuntimeError("Limited-step jobs must be labelled --probe")
    data={};provenance={}
    for split in ["train","val"]:
        metadata=json.loads((features/f"{split}_manifest.json").read_text())
        if not metadata["complete"]:raise RuntimeError("Cannot train from an incomplete feature cache")
        if not a.probe and metadata["config"]["scope"]!="full":raise RuntimeError("Formal training requires full caches")
        expected=(82783,414113) if split=="train" else (40504,202654)
        config=metadata["config"]
        if not a.probe and (config["images"],config["captions"])!=expected:raise RuntimeError("Full sample counts differ")
        if not a.probe:
            hashes=metadata.get("arrays_sha256",{})
            if set(hashes)!={"visual","text","image_index","image_ids","caption_ids"}:raise RuntimeError("Full arrays have no verified hashes")
            for key,digest in hashes.items():
                if sha256(features/f"{split}_{key}.npy")!=digest:raise RuntimeError("Feature array content changed")
        provenance[split]=sha256(features/f"{split}_manifest.json")
        data[split]={key:np.load(features/f"{split}_{key}.npy",mmap_mode="r") for key in ["visual","text","image_index"]}
    n=len(data["train"]["text"]);order_path=root/f"features/sampling/torch_seed{a.seed}_n{n}_e{a.epochs}.npy"
    order_meta=json.loads(order_path.with_suffix(".json").read_text())
    if not order_meta["complete"] or sha256(order_path)!=order_meta["sha256"]:raise RuntimeError("Sampler order is not verified")
    orders=np.load(order_path,mmap_mode="r")
    if orders.shape!=(a.epochs,n):raise RuntimeError("Wrong sampling order shape")
    config={"framework":a.framework,"feature_manifests":provenance,"order_sha256":order_meta["sha256"],
            "initial_weight_sha256":sha256(root/f"checkpoints/mapper_init_seed{a.seed}.npz"),"seed":a.seed,"epochs":a.epochs,"batch_size":128,"lr":1e-4,"optimizer":"Adam: betas=.9,.999,eps=1e-8,wd=0",
            "temperature":.07,"loss":"symmetric CE / actual B squared","alignment":a.alignment,
            "save_best":"minimum COCO validation mean batch loss; no segmentation labels",
            "scope":"probe" if a.probe else "full"}
    out=root/"experiments"/a.run_name;out.mkdir(parents=True,exist_ok=True)
    if (out/"latest.npz").exists() and not a.resume:raise RuntimeError("Run exists; explicitly resume its latest.npz")
    initial=root/f"checkpoints/mapper_init_seed{a.seed}.npz"
    engine=JittorTrainer(initial,a.alignment) if a.framework=="jittor" else TorchTrainer(initial,a.alignment,root)
    state={"epoch":0,"offset":0,"step":0,"epoch_loss_sum":0.,"epoch_batches":0,
           "best_validation":None,"best_epoch":None,"history":[],"seconds":0.}
    if a.resume:
        with np.load(a.resume) as archive:
            saved=json.loads(str(archive["metadata"]))
            if saved["config"]!=config:raise RuntimeError("Resume configuration or feature provenance changed")
            state=saved["state"];engine.restore(archive,state["step"])
    atomic_json(out/"config.json",config)
    stopping=[False]
    def on_stop(signum,frame):stopping[0]=True
    signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    started=time.monotonic();previous_seconds=state["seconds"]
    def checkpoint(path):
        state["seconds"]=previous_seconds+time.monotonic()-started
        first,second,step=engine.state()
        if step!=state["step"]:raise RuntimeError("Optimizer and loop steps differ")
        arrays={"model/"+k:v for k,v in engine.weights().items()}
        arrays.update({"first/"+k:v for k,v in first.items()});arrays.update({"second/"+k:v for k,v in second.items()})
        arrays["metadata"]=np.array(json.dumps({"config":config,"state":state}))
        save_npz(path,arrays)
    def validation():
        losses=[];v=data["val"]
        for offset in range(0,len(v["text"]),128):
            selection=slice(offset,offset+128)
            losses.append(engine.loss(v["visual"][v["image_index"][selection]],v["text"][selection],False))
        return float(np.mean(np.array(losses,np.float32),dtype=np.float32))
    with (out/"train.jsonl").open("a" if a.resume else "w") as log:
        while state["epoch"]<a.epochs:
            epoch=state["epoch"];train=data["train"]
            while state["offset"]<n:
                start=state["offset"];indices=orders[epoch,start:start+128].astype(np.int64)
                loss=engine.loss(train["visual"][train["image_index"][indices]],train["text"][indices],True)
                if not np.isfinite(loss):raise RuntimeError("Non-finite loss")
                state["step"]+=1;state["offset"]+=len(indices)
                state["epoch_loss_sum"]+=loss;state["epoch_batches"]+=1
                if state["step"]==1 or state["step"]%100==0:
                    record={"step":state["step"],"epoch":epoch+1,"offset":state["offset"],"loss":loss,
                            "seconds":previous_seconds+time.monotonic()-started}
                    log.write(json.dumps(record)+"\n");log.flush();print(json.dumps(record),flush=True)
                if state["step"]%a.checkpoint_every==0:checkpoint(out/"latest.npz")
                if stopping[0] or (a.max_steps and state["step"]>=a.max_steps):
                    checkpoint(out/"latest.npz")
                    atomic_json(out/"status.json",{"complete":False,"full_training":not a.probe,"state":state})
                    print("Checkpoint saved at requested stop boundary",flush=True);return
            val=validation()
            record={"epoch":epoch+1,"train_loss":state["epoch_loss_sum"]/state["epoch_batches"],
                    "validation_loss":val,"step":state["step"]}
            if state["best_validation"] is None or val<state["best_validation"]:
                state["best_validation"]=val;state["best_epoch"]=epoch+1
                save_npz(out/"projection_best.npz",engine.weights())
            state["history"].append(record);state["epoch"]+=1;state["offset"]=0
            state["epoch_loss_sum"]=0.;state["epoch_batches"]=0
            checkpoint(out/"latest.npz")
            if state["epoch"]%5==0:checkpoint(out/f"epoch_{state['epoch']:03d}.npz")
            atomic_json(out/"status.json",{"complete":False,"full_training":not a.probe,"state":state})
            print(json.dumps(record),flush=True)
    save_npz(out/"projection_last.npz",engine.weights())
    atomic_json(out/"metrics.json",{"config":config,"complete":True,"full_training":not a.probe,
            "train_images":len(data["train"]["visual"]),"train_captions":n,
            "validation_captions":len(data["val"]["text"]),"state":state})
    atomic_json(out/"status.json",{"complete":True,"full_training":not a.probe,"state":state})
    print("Training complete; final epoch mapper exported for primary evaluation; COCO-best retained for diagnostics",flush=True)
if __name__=="__main__":main()

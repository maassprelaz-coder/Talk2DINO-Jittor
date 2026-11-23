"""Train the mapper on portable pre-extracted features, with resumable Adam state."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import jittor as jt
from .projection import ProjectionLayer,infonce_loss
from .optim import TorchCompatibleAdam


def save_checkpoint(path,model,optimizer,rng,step,epoch,offset,order):
    arrays={f"model/{name}":value.numpy() for name,value in model.named_parameters()}
    for name,first,second in zip(
            [name for name,_ in model.named_parameters()],
            optimizer.param_groups[0]["m"],optimizer.param_groups[0]["values"]):
        arrays["first/"+name]=first.numpy(); arrays["second/"+name]=second.numpy()
    metadata={"step":step,"epoch":epoch,"offset":offset,"optimizer_n_step":optimizer.n_step,
              "rng":rng.bit_generator.state}
    arrays["metadata"]=np.array(json.dumps(metadata))
    arrays["order"]=order
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as f: np.savez(f,**arrays)
    temporary.replace(path)


def load_checkpoint(path,model,optimizer,rng):
    with np.load(path) as archive:
        model.load_parameters({name:jt.array(archive["model/"+name]) for name,_ in model.named_parameters()})
        for name,first,second in zip(
                [name for name,_ in model.named_parameters()],
                optimizer.param_groups[0]["m"],optimizer.param_groups[0]["values"]):
            first.update(jt.array(archive["first/"+name])); second.update(jt.array(archive["second/"+name]))
        metadata=json.loads(str(archive["metadata"]))
        optimizer.n_step=metadata["optimizer_n_step"]
        rng.bit_generator.state=metadata["rng"]
        order=archive["order"].copy()
    return metadata["step"],metadata["epoch"],metadata["offset"],order


def validation(model,visual,texts,image_indices,batch_size):
    losses=[]
    with jt.no_grad():
        for start in range(0,len(texts),batch_size):
            idx=slice(start,start+batch_size)
            loss=infonce_loss(model(jt.array(visual[image_indices[idx]]),jt.array(texts[idx])))
            losses.append(float(loss.numpy().item()))
    return float(np.mean(losses))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--run-name",default="smoke_real_coco_seed123")
    parser.add_argument("--seed",type=int,default=123)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--max-steps",type=int,default=0)
    parser.add_argument("--checkpoint-every",type=int,default=5000)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--alignment",choices=["max_score","mean_heads"],default="max_score")
    args=parser.parse_args(); root=args.root.resolve()
    jt.flags.use_cuda=1
    jt.set_global_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    visual=np.load(args.features/"train_visual.npz")["heads"]
    text_data=np.load(args.features/"train_text.npz")
    text=text_data["text"]; image_index=text_data["image_index"]
    val_visual=np.load(args.features/"val_visual.npz")["average"]
    val_text_data=np.load(args.features/"val_text.npz")
    out=root/"experiments"/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    model=ProjectionLayer(alignment_strategy=args.alignment)
    initial=root/"checkpoints"/f"mapper_init_seed{args.seed}.npz"
    with np.load(initial) as weights:
        model.load_parameters({name:jt.array(weights[name]) for name,_ in model.named_parameters()})
    optimizer=TorchCompatibleAdam(model.parameters(),lr=1e-4)
    step=0; epoch=0; offset=0; order=rng.permutation(len(text))
    if args.resume: step,epoch,offset,order=load_checkpoint(args.resume,model,optimizer,rng)
    initial_val=validation(model,val_visual,val_text_data["text"],val_text_data["image_index"],args.batch_size)
    history=[]; durations=[]
    started=time.perf_counter()
    with (out/"train.jsonl").open("a" if args.resume else "w") as log:
        while epoch<args.epochs and (not args.max_steps or step<args.max_steps):
            if offset>=len(order):
                epoch+=1; offset=0
                if epoch>=args.epochs: break
                order=rng.permutation(len(text))
            indices=order[offset:offset+args.batch_size]
            t0=time.perf_counter()
            loss=infonce_loss(model(jt.array(visual[image_index[indices]]),jt.array(text[indices])))
            loss_value=float(loss.numpy().item())
            optimizer.step(loss)
            jt.sync_all()
            duration=time.perf_counter()-t0
            step+=1; offset+=len(indices)
            history.append(loss_value); durations.append(duration)
            record={"step":step,"epoch":epoch,"batch":len(indices),"loss":loss_value,"seconds":duration}
            log.write(json.dumps(record)+"\n"); log.flush()
            if step==1 or step%10==0: print(json.dumps(record),flush=True)
            if step%args.checkpoint_every==0:
                save_checkpoint(out/f"checkpoint_step{step}.npz",model,optimizer,rng,step,epoch,offset,order)
    final_val=validation(model,val_visual,val_text_data["text"],val_text_data["image_index"],args.batch_size)
    save_checkpoint(out/"checkpoint_final.npz",model,optimizer,rng,step,epoch,offset,order)
    np.savez(out/"projection_final.npz",**{name:value.numpy() for name,value in model.named_parameters()})
    stats={"purpose":"real COCO smoke training; not full paper training or segmentation evaluation",
           "framework":"jittor","steps":step,"epoch_cursor_zero_based":epoch,"epochs_completed":epoch+int(offset>=len(order)),"captions":len(text),
           "images":len(visual),"batch_size":args.batch_size,"seed":args.seed,
           "alignment":args.alignment,"initial_validation_loss":initial_val,"final_validation_loss":final_val,
           "first_10_train_mean":float(np.mean(history[:10])),
           "last_10_train_mean":float(np.mean(history[-10:])),
           "steady_step_seconds":float(np.mean(durations[5:])) if len(durations)>5 else None,
           "total_seconds":time.perf_counter()-started,
           "loss_normalization":"official symmetric CE divided by actual batch size squared"}
    (out/"metrics.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats,indent=2))
    if not np.all(np.isfinite(history)): raise RuntimeError("Non-finite training loss")

if __name__=="__main__": main()

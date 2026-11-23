"""Resumable memory-mapped COCO features; full caches require verified full images."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from talk2dino_jittor.preprocessing import training_image
from talk2dino_jittor.tokenizer import tokenize
from range_download import atomic_json,sha256

COUNTS={"train":(82783,414113),"val":(40504,202654)}

def mapped(path,shape,dtype,exists):
    if exists:
        array=np.load(path,mmap_mode="r+")
        if array.shape!=tuple(shape) or array.dtype!=np.dtype(dtype):
            raise RuntimeError("Cache shape or dtype changed: "+str(path))
        return array
    return np.lib.format.open_memmap(path,mode="w+",dtype=dtype,shape=shape)

class Encoder:
    def __init__(self,root,framework):
        self.root=root; self.framework=framework; self.visual=None; self.text=None
        if framework=="torch":
            import torch
            self.lib=torch
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_num_threads(4)
        else:
            import jittor as jt
            self.lib=jt; jt.flags.use_cuda=1

    def encode_text(self,tokens):
        if self.text is None:
            if self.framework=="torch":
                import clip
                self.text,_=clip.load("ViT-B/16",device="cpu",jit=False,
                    download_root=os.path.join(os.environ.get("TALK2DINO_RUNTIME", os.path.expanduser("~/.local/share/talk2dino")), "clip-cache"))
                self.text=self.text.float().eval().cuda()
            else:
                from talk2dino_jittor.backbones import ClipViTB16Text
                self.text=ClipViTB16Text(self.root/"checkpoints/clip_vitb16_text_fp32.npz")
        with self.lib.no_grad():
            if self.framework=="torch":
                return self.text.encode_text(self.lib.from_numpy(tokens.astype(np.int64)).cuda()).cpu().numpy()
            return self.text(self.lib.array(tokens)).numpy()

    def release_text(self):
        self.text=None
        if self.framework=="torch": self.lib.cuda.empty_cache()
        else: self.lib.gc()

    def encode_image(self,array,split):
        if self.visual is None:
            if self.framework=="torch":
                self.visual=self.lib.hub.load(str(self.root/"upstream/dinov2"),"dinov2_vitb14_reg",
                                              source="local",pretrained=True).float().eval().cuda()
                self.capture={}
                self.visual.blocks[-1].attn.qkv.register_forward_hook(
                    lambda module,inputs,output:self.capture.__setitem__("qkv",output))
            else:
                from talk2dino_jittor.backbones import DinoV2ViTB14Registers
                self.visual=DinoV2ViTB14Registers(self.root/"checkpoints/dinov2_vitb14_reg_fp32.npz")
        with self.lib.no_grad():
            if self.framework=="torch":
                output=self.visual.forward_features(self.lib.from_numpy(array).cuda())
                patches=output["x_norm_patchtokens"]
                qkv=self.capture["qkv"].reshape(len(array),-1,3,12,64).permute(2,0,3,1,4)
                logits=((qkv[0]*.125)@qkv[1].transpose(-2,-1))[:,:,0,5:]
                if split=="train":
                    result=(patches[:,None]*logits.softmax(-1)[:,:,:,None]).mean(2)
                else: result=(patches*logits.mean(1).softmax(-1)[:,:,None]).mean(1)
                return result.cpu().numpy()
            output=self.visual(self.lib.array(array))
            result=output["head_features" if split=="train" else "average_feature"].numpy()
            del output
            self.lib.gc()
            return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--framework",choices=["jittor","torch"],required=True)
    p.add_argument("--scope",choices=["full","probe"],default="full")
    p.add_argument("--stage",choices=["text","visual","all"],default="all")
    p.add_argument("--splits",nargs="+",choices=["train","val"],default=["train","val"])
    p.add_argument("--image-batch",type=int,default=8)
    p.add_argument("--text-batch",type=int,default=128)
    p.add_argument("--max-batches",type=int,default=0,help="Intentional interruption probe; never marks incomplete arrays ready")
    a=p.parse_args();root=a.root.resolve()
    destination=root/f"features/{a.scope}_{a.framework}"
    destination.mkdir(parents=True,exist_ok=True)
    encoder=Encoder(root,a.framework)
    count=0;started=time.monotonic()
    weights={name:sha256(root/"checkpoints"/name) for name in
             ["dinov2_vitb14_reg_fp32.npz","clip_vitb16_text_fp32.npz"]}
    for stage in (["text","visual"] if a.stage=="all" else [a.stage]):
        if stage=="visual":encoder.release_text()
        for split in a.splits:
            ann_path=root/"data/coco2014/annotations"/(
                f"captions_{split}2014.json" if a.scope=="full" else f"smoke_{split}2014_captions.json")
            ann=json.loads(ann_path.read_text());images=ann["images"];captions=ann["annotations"]
            if a.scope=="full" and (len(images),len(captions))!=COUNTS[split]:
                raise RuntimeError("Official full annotation counts differ")
            config={"framework":a.framework,"scope":a.scope,"split":split,"images":len(images),
                    "captions":len(captions),"annotation_sha256":sha256(ann_path),
                    "weights_sha256":weights,"precision":"FP32","image_size":448,
                    "image_preprocessing":"PIL bicubic shortest=448, center crop",
                    "train_visual":"per-head attention softmax pooling / 1024",
                    "val_visual":"mean raw attention logits then softmax pooling / 1024",
                    "sample_order":"original official annotation order"}
            state_path=destination/f"{split}_manifest.json"
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state["config"]!=config:raise RuntimeError("Cache provenance changed")
            else:
                state={"config":config,"text_done":0,"visual_done":0,"complete":False}
                image_index={image["id"]:i for i,image in enumerate(images)}
                if len(image_index)!=len(images):raise RuntimeError("Duplicate COCO image IDs")
                np.save(destination/f"{split}_image_ids.npy",np.array([im["id"] for im in images],np.int64))
                np.save(destination/f"{split}_caption_ids.npy",np.array([cap["id"] for cap in captions],np.int64))
                np.save(destination/f"{split}_image_index.npy",np.array([image_index[cap["image_id"]] for cap in captions],np.int64))
                atomic_json(state_path,state)
            total=len(captions) if stage=="text" else len(images)
            cursor=state[stage+"_done"]
            if cursor==total:
                print(f"{a.framework} {split} {stage} already complete",flush=True);continue
            if stage=="visual" and a.scope=="full":
                preparation=json.loads((root/"data/coco2014/full_preparation.json").read_text())
                if split not in preparation["complete_splits"]:
                    raise RuntimeError("Full image split has not passed archive CRC verification")
            shape=(total,512) if stage=="text" else ((total,12,768) if split=="train" else (total,768))
            target=destination/f"{split}_{stage}.npy"
            values=mapped(target,shape,np.float32,target.exists())
            batch=a.text_batch if stage=="text" else a.image_batch
            last_print=0
            while cursor<total:
                end=min(cursor+batch,total)
                if stage=="text":
                    result=encoder.encode_text(tokenize([cap["caption"] for cap in captions[cursor:end]],truncate=True))
                else:
                    result=encoder.encode_image(np.stack([training_image(root/f"data/coco2014/{split}2014"/im["file_name"])
                                                          for im in images[cursor:end]]),split)
                if not np.all(np.isfinite(result)):raise RuntimeError("Non-finite frozen features")
                values[cursor:end]=result
                cursor=end;count+=1
                stopping=a.max_batches and count>=a.max_batches
                if count%32==0 or cursor==total or stopping:
                    values.flush()
                    state[stage+"_done"]=cursor
                    state["complete"]=(state["text_done"]==len(captions) and state["visual_done"]==len(images))
                    if state["complete"] and a.scope=="full":
                        state["arrays_sha256"]={key:sha256(destination/f"{split}_{key}.npy") for key in
                            ["visual","text","image_index","image_ids","caption_ids"]}
                    atomic_json(state_path,state)
                now=time.monotonic()
                if now-last_print>15 or cursor==total or stopping:
                    print(f"{a.framework} {a.scope} {split} {stage}: {cursor}/{total}; total {now-started:.1f}s",flush=True)
                    last_print=now
                if stopping:
                    print("Stopped at requested batch boundary; incomplete caches remain blocked",flush=True);return
            del values,ann,images,captions
    complete=all((destination/f"{s}_manifest.json").exists() and
                 json.loads((destination/f"{s}_manifest.json").read_text())["complete"] for s in ["train","val"])
    atomic_json(destination/"manifest.json",{"scope":a.scope,"framework":a.framework,"complete":complete,
                "splits":["train","val"],"format":"memory-mapped npy v1"})
    print(json.dumps({"destination":str(destination),"complete":complete,"seconds":time.monotonic()-started}),flush=True)

if __name__=="__main__":main()

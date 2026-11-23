"""Download official backbone weights and export portable FP32 alignment fixtures."""
import argparse
import hashlib
import os
import json
import sys
from pathlib import Path
import numpy as np
import torch
import clip

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--size",type=int,default=448)
    args=parser.parse_args()
    root=args.root.resolve()
    folder=root/"experiments/001_backbone_alignment"
    fixture=folder/"fixtures"
    weights=root/"checkpoints"
    fixture.mkdir(parents=True,exist_ok=True); weights.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    model=torch.hub.load(str(root/"upstream/dinov2"),"dinov2_vitb14_reg",
                         source="local",pretrained=True).float().eval()
    np.savez(weights/"dinov2_vitb14_reg_fp32.npz",
             **{name:value.detach().numpy() for name,value in model.state_dict().items()})
    model=model.cuda()
    input_array=np.random.default_rng(123).standard_normal((1,3,args.size,args.size)).astype(np.float32)
    captures={}
    def save_qkv(module,inputs,output): captures["qkv"]=output.detach()
    handle=model.blocks[-1].attn.qkv.register_forward_hook(save_qkv)
    with torch.no_grad():
        output=model.forward_features(torch.from_numpy(input_array).cuda())
        qkv=captures["qkv"].reshape(1,-1,3,12,64).permute(2,0,3,1,4)
        logits=((qkv[0]*0.125)@qkv[1].transpose(-2,-1))[:,:,0,5:]
        patches=output["x_norm_patchtokens"]
        head_features=(patches[:,None]*logits.softmax(-1)[:,:,:,None]).mean(dim=2)
        avg_feature=(patches*logits.mean(dim=1).softmax(-1)[:,:,None]).mean(dim=1)
        resized_pos=model.interpolate_pos_encoding(
            torch.empty(1,(args.size//14)**2+1,768,device="cuda"),args.size,args.size)
    expected={"patches":patches.cpu().numpy(),"cls":output["x_norm_clstoken"].cpu().numpy(),
              "attention_logits":logits.cpu().numpy(),"head_features":head_features.cpu().numpy(),
              "average_feature":avg_feature.cpu().numpy(),
              "position_embedding":resized_pos.cpu().numpy()}
    np.savez(fixture/f"dino_{args.size}_expected.npz",**expected)
    np.save(fixture/f"image_{args.size}.npy",input_array)
    handle.remove()
    del model,output,patches,qkv,logits,head_features,avg_feature
    torch.cuda.empty_cache()
    text_model,_=clip.load("ViT-B/16",device="cpu",jit=False,
                           download_root=os.path.join(os.environ.get("TALK2DINO_RUNTIME", os.path.expanduser("~/.local/share/talk2dino")), "clip-cache"))
    text_model=text_model.float().eval()
    text_state={name:value.detach().numpy() for name,value in text_model.state_dict().items()
                if not name.startswith("visual.") and name!="logit_scale"}
    np.savez(weights/"clip_vitb16_text_fp32.npz",**text_state)
    texts=["a photo of a cat","a traffic sign","a forest"]
    tokens=clip.tokenize(texts)
    text_model=text_model.cuda()
    with torch.no_grad():
        text_features=text_model.encode_text(tokens.cuda()).float().cpu().numpy()
    np.save(fixture/"text_tokens.npy",tokens.numpy())
    np.save(fixture/"text_expected.npy",text_features)
    metadata={"torch":torch.__version__,"precision":"float32, TF32 disabled",
              "image_size":args.size,"texts":texts,
              "models":{"dino":"dinov2_vitb14_reg","clip_text":"ViT-B/16"},
              "weight_files":{p.name:sha256_file(p) for p in weights.glob("*.npz")},
              "test_type":"synthetic_image_tensor_and_real_text_numerical_alignment"}
    (folder/f"reference_{args.size}.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))

if __name__=="__main__": main()

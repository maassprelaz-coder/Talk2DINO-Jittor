"""Compare Jittor frozen encoders to actual official-model reference tensors."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import jittor as jt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from talk2dino_jittor.backbones import DinoV2ViTB14Registers,ClipViTB16Text
from talk2dino_jittor.interpolation import resize_position_embedding
from talk2dino_jittor.tokenizer import tokenize

def check(name,actual,expected,results,atol=1e-4,rtol=1e-4):
    actual,expected=np.asarray(actual),np.asarray(expected)
    difference=np.abs(actual-expected)
    cosine=float(np.sum(actual.astype(np.float64)*expected)/(
        np.linalg.norm(actual.astype(np.float64))*np.linalg.norm(expected.astype(np.float64))))
    results[name]={"pass":bool(np.allclose(actual,expected,atol=atol,rtol=rtol)),
                   "max_abs":float(difference.max()),"mean_abs":float(difference.mean()),
                   "cosine":cosine,"atol":atol,"rtol":rtol}
    print(name,json.dumps(results[name]),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--size",type=int,default=448)
    parser.add_argument("--model",choices=["dino","clip","both"],default="both")
    args=parser.parse_args(); root=args.root.resolve()
    jt.flags.use_cuda=1
    folder=root/"experiments/001_backbone_alignment"
    fixture=folder/"fixtures"
    results={}
    if args.model in ["dino","both"]:
        with jt.no_grad():
            model=DinoV2ViTB14Registers(root/"checkpoints/dinov2_vitb14_reg_fp32.npz")
            expected=np.load(fixture/f"dino_{args.size}_expected.npz")
            check("position_embedding",resize_position_embedding(model.original_position,args.size//14,args.size//14),
                  expected["position_embedding"],results,atol=1e-6,rtol=1e-5)
            if not results["position_embedding"]["pass"]:
                raise RuntimeError("Position interpolation failed before backbone comparison")
            result=model(jt.array(np.load(fixture/f"image_{args.size}.npy")))
            for name,value in result.items():
                check("dino/"+name,value.numpy(),expected[name],results)
            del model,result
            jt.gc()
    if args.model in ["clip","both"]:
        with jt.no_grad():
            metadata=json.loads((folder/f"reference_{args.size}.json").read_text())
            tokens=tokenize(metadata["texts"])
            check("tokens",tokens,np.load(fixture/"text_tokens.npy"),results,atol=0,rtol=0)
            model=ClipViTB16Text(root/"checkpoints/clip_vitb16_text_fp32.npz")
            output=model(jt.array(tokens)).numpy()
            check("clip/text",output,np.load(fixture/"text_expected.npy"),results)
    report={"pass":all(x["pass"] for x in results.values()),"device":"cuda","jittor":jt.__version__,
            "size":args.size,"checks":results,
            "test_type":"synthetic_image_tensor_and_real_text_numerical_alignment"}
    (folder/f"alignment_{args.model}_{args.size}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["pass"]: raise SystemExit(1)

if __name__=="__main__": main()

"""Stream full caches in bounded memory; verify IDs and frozen feature parity."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
records=[]
for split in ["train","val"]:
    folders=[a.root/f"features/full_{f}" for f in ["jittor","torch"]]
    for folder in folders:
        if not json.loads((folder/f"{split}_manifest.json").read_text())["complete"]:raise RuntimeError("Incomplete full cache")
    for key in ["image_ids","caption_ids","image_index"]:
        if not np.array_equal(*[np.load(folder/f"{split}_{key}.npy",mmap_mode="r") for folder in folders]):
            raise RuntimeError("Cross-framework sample identity differs")
    for kind in ["visual","text"]:
        left,right=[np.load(folder/f"{split}_{kind}.npy",mmap_mode="r") for folder in folders]
        maximum=0.;minimum_cos=1.;bad=0;elements=0
        for start in range(0,len(left),256):
            x=np.asarray(left[start:start+256]);y=np.asarray(right[start:start+256])
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):raise RuntimeError("Non-finite cache")
            maximum=max(maximum,float(np.max(abs(x-y))))
            bad+=int(np.count_nonzero(~np.isclose(x,y,atol=1e-5,rtol=1e-4)));elements+=x.size
            xd=x.astype(np.float64);yd=y.astype(np.float64)
            cosine=np.sum(xd*yd,axis=-1)/np.sqrt(np.sum(xd*xd,axis=-1)*np.sum(yd*yd,axis=-1))
            minimum_cos=min(minimum_cos,float(cosine.min()))
        record={"split":split,"kind":kind,"rows":len(left),"elements":elements,"max_abs":maximum,
                "min_cosine":minimum_cos,"elements_outside_atol1e5_rtol1e4":bad,"pass":bad==0}
        records.append(record);print(json.dumps(record),flush=True)
result={"checks":records,"pass":all(x["pass"] for x in records)}
out=a.root/"experiments/004_full_data/full_feature_alignment.json"
out.write_text(json.dumps(result,indent=2)+"\n")
if not result["pass"]:raise RuntimeError("Full-cache parity requires investigation; formal training not started")

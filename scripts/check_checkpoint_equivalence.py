"""Compare uninterrupted and resumed actual training, including Adam and RNG states."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,required=True)
args=parser.parse_args(); root=args.root.resolve()
full=np.load(root/"experiments/smoke_real_coco_seed123/checkpoint_final.npz")
resumed=np.load(root/"experiments/smoke_real_coco_resume50_seed123/checkpoint_final.npz")
checks={}
for key in full.files:
    if key=="metadata":
        checks[key]={"pass":json.loads(str(full[key]))==json.loads(str(resumed[key]))}
    elif key=="order":
        checks[key]={"pass":bool(np.array_equal(full[key],resumed[key]))}
    else:
        difference=np.abs(full[key]-resumed[key])
        checks[key]={"pass":bool(np.allclose(full[key],resumed[key],atol=1e-7,rtol=1e-6)),
                     "max_abs":float(difference.max()),"mean_abs":float(difference.mean())}
report={"pass":all(value["pass"] for value in checks.values()),"checks":checks,
        "scope":"100-step real COCO run versus resuming from the exact step-50 checkpoint"}
out=root/"experiments/002_real_coco_smoke"
out.mkdir(exist_ok=True)
(out/"checkpoint_equivalence.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
if not report["pass"]: raise SystemExit(1)

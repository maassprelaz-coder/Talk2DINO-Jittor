"""Choose the official final-epoch checkpoint independently of downstream metrics."""
import json
from pathlib import Path
import numpy as np
import yaml
from range_download import atomic_json,sha256

def primary_mapper(root,run):
    root=Path(root);run=Path(run)
    official=root/"upstream/Talk2DINO/configs/vitb_mlp_infonce.yaml"
    config=yaml.safe_load(official.read_text())
    if config["train"].get("save_best_model") is not False:
        raise RuntimeError("Pinned official checkpoint-selection policy changed")
    result=json.loads((run/"metrics.json").read_text())
    state=result["state"]
    if not result.get("complete") or not result.get("full_training") or state["epoch"]!=100:
        raise RuntimeError("Primary evaluation requires a completed 100-epoch full run")
    final=run/"projection_last.npz"
    with np.load(final) as weights,np.load(run/"latest.npz") as latest:
        metadata=json.loads(str(latest["metadata"]))
        if metadata["state"]["epoch"]!=100 or metadata["state"]["step"]!=state["step"]:
            raise RuntimeError("Final export and checkpoint metadata differ")
        names={name.removeprefix("model/") for name in latest.files if name.startswith("model/")}
        if set(weights.files)!=names:raise RuntimeError("Final mapper parameter names differ")
        if not all(np.array_equal(weights[name],latest["model/"+name]) for name in names):
            raise RuntimeError("Final mapper is not the last checkpoint's exact parameter export")
    atomic_json(run/"primary_selection.json",{
        "policy":"official save_best_model=false: evaluate final epoch 100",
        "official_config_sha256":sha256(official),"primary_mapper":final.name,
        "primary_mapper_sha256":sha256(final),"epochs":100,"steps":state["step"],
        "verified_equal_to_latest_checkpoint":True,
        "supplementary_mapper":"projection_best.npz",
        "supplementary_selection":"COCO minimum validation loss; not used for primary result",
        "supplementary_epoch":state["best_epoch"]})
    return final

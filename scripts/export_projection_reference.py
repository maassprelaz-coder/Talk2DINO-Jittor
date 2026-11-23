"""Execute the exact official classes and export framework-neutral test fixtures."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml


def official_class(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in classes} != set(names):
        raise RuntimeError("Official class definitions changed")
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = root / "upstream/Talk2DINO"
    out = root / "experiments/000_projection_alignment/fixtures"
    out.mkdir(parents=True, exist_ok=True)
    namespace = {"torch": torch, "nn": nn, "F": F, "yaml": yaml, "np": np,
                 "CLIPLastLayer": type("UnusedCLIPLastLayer", (), {})}
    official_class(upstream / "src/model.py", ["ProjectionLayer"], namespace)
    official_class(upstream / "src/loss.py", ["Contrastive", "ContrastiveLoss"], namespace)
    Projection = namespace["ProjectionLayer"]
    Criterion = namespace["ContrastiveLoss"]
    model = Projection(act=nn.Tanh(), hidden_layer=True, dino_embed_dim=768)
    official_weight = upstream / "weights/vitb_mlp_infonce.pth"
    model.load_state_dict(torch.load(official_weight, map_location="cpu", weights_only=True))
    state = {name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}
    np.savez(out / "weights.npz", **state)
    rng = np.random.default_rng(123)
    visual = rng.standard_normal((8, 12, 768)).astype(np.float32)
    text = rng.standard_normal((8, 512)).astype(np.float32)
    criterion = Criterion(model, ltype="infonce")
    model.train()
    visual_t, text_t = torch.from_numpy(visual), torch.from_numpy(text)
    scores, indices = model(visual_t, text_t, return_index=True)
    loss = criterion(visual_t, text_t)
    projected = model.project_clip_txt(text_t)
    loss.backward()
    grads = {name: tensor.grad.detach().numpy() for name, tensor in model.named_parameters()}
    expected = {"scores": scores.detach().numpy(), "head_indices": indices.detach().numpy(),
                "projected_text": projected.detach().numpy(),
                "loss": loss.detach().numpy()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    updated = {name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}
    np.savez(out / "inputs.npz", visual=visual, text=text)
    np.savez(out / "expected.npz", **expected)
    np.savez(out / "gradients.npz", **grads)
    np.savez(out / "adam_step.npz", **updated)
    metadata = {
        "test_type": "synthetic_tensor_numerical_alignment_not_paper_experiment",
        "upstream_commit": "1462d0ebbf3c8d2dab623a03d41548c8e6098f34",
        "official_weight_sha256": hashlib.sha256(official_weight.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
                          for name in ["src/model.py", "src/loss.py"]},
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "seed": 123, "shape_visual": list(visual.shape), "shape_text": list(text.shape),
        "official_loss_includes_batch_squared_divisor": True,
        "temperature_updated_by_reference_optimizer": False,
        "reference_loss": float(loss.detach()),
    }
    (out.parent / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()

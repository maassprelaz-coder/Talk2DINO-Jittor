"""Check Jittor forward, backward and Adam against exported official PyTorch fixtures."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import jittor as jt
from jittor import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talk2dino_jittor.projection import ProjectionLayer, infonce_loss
from talk2dino_jittor.optim import TorchCompatibleAdam


def compare(name, actual, expected, report, atol=1e-5, rtol=1e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    diff = np.abs(actual - expected)
    ok = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    report[name] = {"pass": ok, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                    "atol": atol, "rtol": rtol}
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    root = args.root.resolve()
    folder = root / "experiments/000_projection_alignment"
    fixture = folder / "fixtures"
    model = ProjectionLayer()
    weights = dict(np.load(fixture / "weights.npz"))
    model.load_parameters({name: jt.array(value) for name, value in weights.items()})
    data = np.load(fixture / "inputs.npz")
    expected = np.load(fixture / "expected.npz")
    visual, text = jt.array(data["visual"]), jt.array(data["text"])
    scores, heads = model(visual, text, return_index=True)
    loss = infonce_loss(scores)
    report = {}
    passed = [
        compare("projected_text", model.project_clip_txt(text).numpy(), expected["projected_text"], report),
        compare("scores", scores.numpy(), expected["scores"], report),
        compare("head_indices", heads.numpy(), expected["head_indices"], report, atol=0, rtol=0),
        compare("loss", loss.numpy().reshape(()), expected["loss"], report, atol=1e-7, rtol=1e-4),
    ]
    grad_ref = np.load(fixture / "gradients.npz")
    parameters = list(model.named_parameters())
    gradients = jt.grad(loss, [value for _, value in parameters])
    for (name, _), gradient in zip(parameters, gradients):
        passed.append(compare("gradient/" + name, gradient.numpy(), grad_ref[name], report,
                              atol=1e-6, rtol=1e-3))
    optimizer = TorchCompatibleAdam(model.parameters(), lr=1e-4)
    optimizer.step(infonce_loss(model(visual, text)))
    updated_ref = np.load(fixture / "adam_step.npz")
    for name, parameter in model.named_parameters():
        passed.append(compare("adam/" + name, parameter.numpy(), updated_ref[name], report))
    payload = {"pass": all(passed), "jittor": jt.__version__, "device": args.device,
               "test_type": "synthetic_tensor_numerical_alignment_not_paper_experiment",
               "checks": report}
    target = folder / ("alignment_" + args.device + ".json")
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

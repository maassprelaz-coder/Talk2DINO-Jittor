"""Jittor port of the official text projection and paired head selection.

Reference: lorebianchi98/Talk2DINO, commit 1462d0e, src/model.py.
Only the ICCV ViT-B sentence-level text / per-head visual feature path is
implemented here. Alternative token-level research variants are not implied.
"""
import math
import jittor as jt
from jittor import nn
from .runtime import configure_fp32
configure_fp32()


def l2_normalize(x, dim=-1, eps=1e-12):
    denominator = jt.sqrt((x * x).sum(dim=dim, keepdims=True))
    return x / jt.maximum(denominator, eps)


class AlignedLinear(nn.Linear):
    """Use explicit FP32 cuBLAS for dense GPU layers, preserving state names."""
    def execute(self, x):
        if jt.flags.use_cuda and jt.compile_extern.cublas_ops:
            shape=tuple(x.shape[:-1])+(self.weight.shape[0],)
            matrix=x.reshape((1,-1,self.weight.shape[1]))
            result=jt.compile_extern.cublas_ops.cublas_batched_matmul(
                matrix,self.weight.unsqueeze(0),0,1).reshape(shape)
            return result+self.bias if self.bias is not None else result
        return super().execute(x)


class ProjectionLayer(nn.Module):
    def __init__(self, clip_embed_dim=512, dino_embed_dim=768,
                 hidden_layer=True, alignment_strategy="max_score"):
        self.linear_layer = AlignedLinear(clip_embed_dim, dino_embed_dim)
        hidden_count = 1 if hidden_layer is True else int(hidden_layer or 0)
        self.hidden_layers = nn.ModuleList(
            [AlignedLinear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_count)]
        )
        if alignment_strategy not in ("max_score", "mean_heads"):
            raise ValueError("Supported strategies: max_score, mean_heads")
        self.alignment_strategy = alignment_strategy

    def project_clip_txt(self, text):
        x = self.linear_layer(text.float32())
        for layer in self.hidden_layers:
            x = layer(jt.tanh(x))
        return x

    def execute(self, visual, text, return_index=False, ret_embeds=False):
        text = l2_normalize(self.project_clip_txt(text))
        if self.alignment_strategy == "mean_heads" and visual.ndim == 3:
            # Explicit ablation: uniform pooling of head vectors BEFORE L2.
            visual = visual.mean(dim=1)
        visual = l2_normalize(visual.float32())
        if ret_embeds:
            return text, visual
        indices = None
        if visual.ndim == 3:
            if text.shape[0] != visual.shape[0]:
                raise ValueError("Head selection requires paired image/text batches")
            paired_scores = (text.unsqueeze(1) * visual).sum(dim=-1)
            indices = nn.softmax(paired_scores, dim=-1).argmax(dim=-1)[0]
            visual = visual[jt.arange(visual.shape[0]), indices]
        elif visual.ndim != 2:
            raise ValueError("Visual features must have shape [B,D] or [B,H,D]")
        scores = text @ visual.transpose(1, 0)
        if return_index:
            return scores, indices
        return scores


def infonce_loss(scores, temperature=0.07, official_batch_normalization=True):
    """Match the official symmetric CE, including its extra B**2 divisor.

    The reference optimizer updates only the projection model, not the
    criterion's logit_scale Parameter, so the temperature is fixed at 0.07.
    The returned loss is NOT the unscaled formula printed in the paper.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("InfoNCE expects a square paired similarity matrix")
    scale = jt.exp(jt.array([math.log(1.0 / temperature)]).float32()).stop_grad()
    logits = scale * scores
    labels = jt.arange(scores.shape[0]).int32()
    loss = (nn.cross_entropy_loss(logits, labels)
            + nn.cross_entropy_loss(logits.transpose(1, 0), labels)) / 2
    if official_batch_normalization:
        loss = loss / (scores.shape[0] ** 2)
    return loss

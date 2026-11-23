"""NumPy-only segmentation metrics; no framework-specific label conversion."""
import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        keep = target != self.ignore_index
        gt = target[keep].astype(np.int64)
        pred = prediction[keep].astype(np.int64)
        if np.any((gt < 0) | (gt >= self.num_classes)):
            raise ValueError("Ground-truth label outside class range")
        if np.any((pred < 0) | (pred >= self.num_classes)):
            raise ValueError("Predicted label outside class range")
        encoded = gt * self.num_classes + pred
        self.matrix += np.bincount(
            encoded, minlength=self.num_classes ** 2
        ).reshape(self.matrix.shape)

    def compute(self):
        tp = np.diag(self.matrix).astype(np.float64)
        union = self.matrix.sum(0) + self.matrix.sum(1) - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union != 0)
        mean = float(np.nanmean(iou)) if np.any(union != 0) else float("nan")
        return {"miou": mean, "iou": iou.tolist(), "confusion": self.matrix.tolist()}

"""Adam whose epsilon placement matches the official PyTorch optimizer."""
import math
import jittor as jt
from jittor.optim import Adam as NativeAdam


class TorchCompatibleAdam(NativeAdam):
    """Preserve PyTorch Adam's sqrt(v / bias_correction2) + eps denominator.

    Jittor 1.3.11 native Adam places eps before bias correction. That difference
    matters for the small gradients caused by the official B**2 loss divisor.
    This subclass keeps Jittor's optimizer state and gradient machinery.
    """
    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        step = float(self.n_step)
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            eps = group.get("eps", self.eps)
            decay = group.get("weight_decay", self.weight_decay)
            beta1, beta2 = group.get("betas", self.betas)
            correction1 = 1.0 - beta1 ** step
            correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
            for parameter, gradient, second, first in zip(
                    group["params"], group["grads"], group["values"], group["m"]):
                if parameter.is_stop_grad():
                    continue
                gradient = gradient + decay * parameter
                first.update(beta1 * first + (1.0 - beta1) * gradient)
                second.update(beta2 * second + (1.0 - beta2) * gradient * gradient)
                denominator = jt.sqrt(second) / correction2_sqrt + eps
                parameter.update(parameter - (lr / correction1) * first / denominator)
        self.post_step()

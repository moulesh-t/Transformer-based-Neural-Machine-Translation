import torch
import torch.optim as optim
import torch.nn as nn
from collections.abc import Callable

class AdamW(optim.Optimizer):
    """
    Implementing the AdamW Optimizer by subclassing the base Optimizer
    """
    
    def __init__(self, params: list[nn.Parameter], lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None: #type:ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # First moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2**0.5)
                ).add_(eps)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return None
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr: float=1e-3, betas:tuple[float,float]=(0.9,0.999),eps:float=1e-8,weight_decay:float=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # get params lr,betas...
            for p in group["params"]:
                if p.grad is None:
                    continue

                # get grad
                grad = p.grad.data

                # get state
                state = self.state[p]

                # initial states
                if len(state) == 0:
                    state["step"] = 0
                    # First moment
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Second moment
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # read state
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # upgrad first moment(momentum)
                # m = beta1 * m + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)

                # upgrad second moment(square of gradient)
                # v = beta2 * v + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # compute bias correction
                # alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                # step_size mul the grad
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                # upgrad parameter
                # theta = theta - step.size * m / (sqrt(v) * eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # weight decay
                # theta = theta - lr * lambda * theta
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr*weight_decay)
                
        return loss


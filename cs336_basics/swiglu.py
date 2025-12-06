import torch
import torch.nn as nn
from cs336_basics.Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            hidden_dim = int(d_model * 8/3)
            self.hidden_dim = (hidden_dim + 63) // 64 * 64
        self.hidden_dim = d_ff
        self.w_gate = Linear(d_model, self.hidden_dim, device=device, dtype=dtype)
        self.w_in = Linear(d_model, self.hidden_dim, device=device, dtype=dtype)
        self.w_out = Linear(self.hidden_dim, d_model,device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # uproject input to gate and value
        gate = self.w_gate(x)
        val = self.w_in(x)

        # computr activated Gate
        gate_act = gate * torch.sigmoid(gate)

        # gateed value
        combined = gate_act * val

        return self.w_out(combined)

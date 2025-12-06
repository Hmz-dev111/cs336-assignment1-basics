import torch
import torch.nn as nn
from cs336_basics.multiheadattention import MultiHeadAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm import Rmsnorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.rms_norm1 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.rms_norm2 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # First sub-layer: Multi-Head Attention with Residual Connection
        x_normed1 = self.rms_norm1(x)
        attn_output = self.attn(x_normed1, token_positions)
        h = x + attn_output

        # Second sub-layer: Feed-Forward Network with Residual Connection
        x_normed2 = self.rms_norm2(h)
        ffn_output = self.ffn(x_normed2)
        out = h + ffn_output

        return out
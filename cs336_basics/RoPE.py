import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        # Precompute the sinusoidal frequencies
        freqs = 1.0/(theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype).float() / d_k))
        # location indices
        t = torch.arange(max_seq_len, device=freqs.device)

        # Compute the outer product of location indices and frequencies
        freqs = torch.outer(t, freqs)

        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        # Register as buffer to avoid being treated as a parameter
        self.register_buffer("cos_cached",emb.cos())
        self.register_buffer("sin_cached",emb.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        return (x * cos) + (rotate_half(x)*sin)




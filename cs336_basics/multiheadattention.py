import torch
import torch.nn as nn
from cs336_basics.Linear import Linear
from cs336_basics.RoPE import RoPE
from jaxtyping import Bool, Float, Int
from torch import Tensor

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(in_features - x_max)
    x_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / x_sum

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # get d_k
    d_k = Q.shape[-1]
    # compute raw attention scores
    scores = torch.matmul(Q, K.transpose(-1, -2))
    # scale scores
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype))

    # apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == True, -float('inf'))

    # compute softmax
    attn_probs = run_softmax(scores, dim=-1)

    attn_output = torch.matmul(attn_probs, V)
    return attn_output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_heads

        # comput head dimension
        self.head_dim = d_model // num_heads

        # define Q, K, V, O projection layers
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)

        # define output projection layer
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

        # initialize RoPE
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(d_k = self.head_dim, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        # compute Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # reshape Q, K, V to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)

        # apply RoPE to Q and K
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # create casual mask
        mask = torch.triu(torch.ones((seq_len, seq_len),device=x.device, dtype=torch.bool), diagonal=1)

        # compute attention
        attn_out = run_scaled_dot_product_attention(q, k, v, mask=mask)

        # merge heads and project output
        attn_out = attn_out.transpose(1,2).reshape(batch_size, seq_len, self.d_model)

        return self.w_o(attn_out)
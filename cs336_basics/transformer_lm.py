import torch
import torch.nn as nn
from cs336_basics.embedding import Ebedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import Rmsnorm
from cs336_basics.Linear import Linear

class transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        # token embedding
        self.token_embedding = Ebedding(vocab_size, d_model, device=device, dtype=dtype)

        # tranformer blocks list
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        # final layer norm(RMSNorm)
        self.ln_f = Rmsnorm(d_model, device=device, dtype=dtype)

        # output projection layer(Linear)
        self.lm_head = Linear(d_model,vocab_size, device=device, dtype=dtype)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # create token positions
        seq_len = indices.shape[1]
        token_positions = torch.arange(seq_len, device=indices.device)

        # get token embeddings
        X = self.token_embedding(indices)

        # pass through transformer blocks
        for layer in self.layers:
            X = layer(X, token_positions)

        # final layer norm
        X = self.ln_f(X)

        # output projection to vocab size
        logits = self.lm_head(X)

        return logits

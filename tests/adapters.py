from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Iterable

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.get_tokenizer import Tokenizer
from cs336_basics.Linear import Linear
from cs336_basics.embedding import Ebedding
from cs336_basics.rmsnorm import Rmsnorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.RoPE import RoPE
from cs336_basics.multiheadattention import MultiHeadAttention
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.transformer_lm import transformer_lm
from cs336_basics.AdamW import AdamW
import torch.nn as nn
import math
import numpy as np


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Linear:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    linear = Linear(d_in, d_out)
    with torch.inference_mode():
        linear.weight = nn.Parameter(weights.T)

    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Ebedding:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    Ebedding_layer = Ebedding(vocab_size, d_model)
    with torch.inference_mode():
        Ebedding_layer.weight = nn.Parameter(weights)

    return Ebedding_layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> SwiGLU:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    SwiGLU_layer = SwiGLU(d_model, d_ff)
    with torch.inference_mode():
        SwiGLU_layer.w_gate.weight = nn.Parameter(w1_weight.T)
        SwiGLU_layer.w_in.weight = nn.Parameter(w3_weight.T)
        SwiGLU_layer.w_out.weight = nn.Parameter(w2_weight.T)

    return SwiGLU_layer(in_features)
    


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
        scores = scores.masked_fill(mask == False, -float('inf'))

    # compute softmax
    attn_probs = run_softmax(scores, dim=-1)

    attn_output = torch.matmul(attn_probs, V)
    return attn_output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> MultiHeadAttention:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MultiHeadAttention_layer = MultiHeadAttention(d_model, num_heads,max_seq_len= in_features.shape[-2], device=in_features.device, dtype=in_features.dtype)

    with torch.inference_mode():
        MultiHeadAttention_layer.w_q.weight = nn.Parameter(q_proj_weight.T)
        MultiHeadAttention_layer.w_k.weight = nn.Parameter(k_proj_weight.T)
        MultiHeadAttention_layer.w_v.weight = nn.Parameter(v_proj_weight.T)
        MultiHeadAttention_layer.w_o.weight = nn.Parameter(o_proj_weight.T)
    return MultiHeadAttention_layer(in_features, token_positions=None)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> MultiHeadAttention:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MultiHeadAttention_layer = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=in_features.device, dtype=in_features.dtype) 
    with torch.inference_mode():
        MultiHeadAttention_layer.w_q.weight = nn.Parameter(q_proj_weight.T)
        MultiHeadAttention_layer.w_k.weight = nn.Parameter(k_proj_weight.T)
        MultiHeadAttention_layer.w_v.weight = nn.Parameter(v_proj_weight.T)
        MultiHeadAttention_layer.w_o.weight = nn.Parameter(o_proj_weight.T)
    return MultiHeadAttention_layer(in_features, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> RoPE:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    RoPE_layer = RoPE(theta, d_k, max_seq_len)
    return RoPE_layer(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> TransformerBlock:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    seq_len = in_features.shape[1]
    token_positions = torch.arange(seq_len, device=in_features.device)

    transformer_block_layer = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=in_features.device, dtype=in_features.dtype)
    with torch.inference_mode():
        transformer_block_layer.attn.w_q.weight = nn.Parameter(weights["attn.q_proj.weight"].T)
        transformer_block_layer.attn.w_k.weight = nn.Parameter(weights["attn.k_proj.weight"].T)
        transformer_block_layer.attn.w_v.weight = nn.Parameter(weights["attn.v_proj.weight"].T)
        transformer_block_layer.attn.w_o.weight = nn.Parameter(weights["attn.output_proj.weight"].T)

        transformer_block_layer.rms_norm1.weight = nn.Parameter(weights["ln1.weight"])

        transformer_block_layer.ffn.w_gate.weight = nn.Parameter(weights["ffn.w1.weight"].T)
        transformer_block_layer.ffn.w_in.weight = nn.Parameter(weights["ffn.w3.weight"].T)
        transformer_block_layer.ffn.w_out.weight = nn.Parameter(weights["ffn.w2.weight"].T)
        
        transformer_block_layer.rms_norm2.weight = nn.Parameter(weights["ln2.weight"])

    return transformer_block_layer(in_features, token_positions=token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm_layer = transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=in_indices.device, dtype=torch.float32)
    with torch.inference_mode():
        # embedding
        transformer_lm_layer.token_embedding.weight = nn.Parameter(weights["token_embeddings.weight"])

        # for every num_layers
        for i in range(num_layers):
            prefix = f"layers.{i}."
            block = transformer_lm_layer.layers[i]

            block.attn.w_q.weight = nn.Parameter(weights[f"{prefix}attn.q_proj.weight"].T)
            block.attn.w_k.weight = nn.Parameter(weights[f"{prefix}attn.k_proj.weight"].T)
            block.attn.w_v.weight = nn.Parameter(weights[f"{prefix}attn.v_proj.weight"].T)
            block.attn.w_o.weight = nn.Parameter(weights[f"{prefix}attn.output_proj.weight"].T)

            block.rms_norm1.weight = nn.Parameter(weights[f"{prefix}ln1.weight"])

            block.ffn.w_gate.weight = nn.Parameter(weights[f"{prefix}ffn.w1.weight"].T)
            block.ffn.w_in.weight = nn.Parameter(weights[f"{prefix}ffn.w3.weight"].T)
            block.ffn.w_out.weight = nn.Parameter(weights[f"{prefix}ffn.w2.weight"].T)
            
            block.rms_norm2.weight = nn.Parameter(weights[f"{prefix}ln2.weight"])

        # final norm
        transformer_lm_layer.ln_f.weight = nn.Parameter(weights["ln_final.weight"])

        # output(linear)
        transformer_lm_layer.lm_head.weight = nn.Parameter(weights["lm_head.weight"].T)

    return transformer_lm_layer(in_indices)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Rmsnorm:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    Rmsnorm_layer = Rmsnorm(d_model, eps)
    with torch.inference_mode():
        Rmsnorm_layer.scale = nn.Parameter(weights)

    return Rmsnorm_layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


# def run_get_batch(
#     dataset: npt.NDArray, batch_size: int, context_length: int, device: str
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Given a dataset (a 1D numpy array of integers) and a desired batch size and
#     context length, sample language modeling input sequences and their corresponding
#     labels from the dataset.

#     Args:
#         dataset (np.array): 1D numpy array of integer token IDs in the dataset.
#         batch_size (int): Desired batch size to sample.
#         context_length (int): Desired context length of each sampled example.
#         device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
#             to place the sampled input sequences and labels on.

#     Returns:
#         Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
#         is the sampled input sequences, and the second tuple item is the corresponding
#         language modeling labels.
#     """
#     # product batch_size indices
#     ix = torch.randint(low=0, high=len(dataset)-context_length, size=(batch_size,))

#     # get data (input,x)
#     # x = torch.stack([
#     #     torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
#     #     for i in ix
#     # ])

#     # # get target (target,y)
#     # y = torch.stack([
#     #     torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
#     #     for i in ix
#     # ])

#     #   # device
#     # if "cuda" in device:
#     #     x = x.pin_memory().to(device, non_blocking=True)
#     #     y = y.pin_memory().to(device, non_blocking=True)
#     # else:
#     #     x = x.to(device)
#     #     y = y.to(device)

#     # numpy
#     # batch_x = []
#     # batch_y = []
#     # for i in ix:
#     #     batch_x.append(dataset[i : i + context_length])
#     #     batch_y.append(dataset[i + 1 : i + 1 + context_length])

#     # x = torch.tensor(np.array(batch_x), dtype=torch.long).to(device)
#     # y = torch.tensor(np.array(batch_y), dtype=torch.long).to(device)

#     # return x,y

#     indices = ix.view(-1, 1) + torch.arange(context_length + 1)
#     indices = indices.numpy() # 转回 numpy 去切片
    
#     # 3. 一次性从 dataset 切片 (这里 dataset 必须在内存里，即 np.fromfile 读出来的)
#     batch_data = dataset[indices] # 形状: [Batch, Context+1]
    
#     # 4. 转为 Tensor 并切分 x, y
#     batch_tensor = torch.from_numpy(batch_data.astype(np.int64))
    
#     x = batch_tensor[:, :-1] # 前 context_length 个
#     y = batch_tensor[:, 1:]  # 后 context_length 个
#     # --- 优化结束 ---

#     # 5. 搬运到 GPU
#     if "cuda" in device:
#         x = x.pin_memory().to(device, non_blocking=True)
#         y = y.pin_memory().to(device, non_blocking=True)
#     else:
#         x = x.to(device)
#         y = y.to(device)

#     return x, y

def run_get_batch(dataset, batch_size, context_length, device):
    # 1. 随机生成起始位置
    ix = torch.randint(low=0, high=len(dataset) - context_length, size=(batch_size,))
    
    # 2. ⚡️ 极速向量化切片 (无 for 循环)
    # 利用广播机制一次性生成所有需要的索引矩阵 [Batch, Context+1]
    # 这一步是在 CPU 上做的纯数学运算，极快
    start_indices = ix.view(-1, 1)
    offsets = torch.arange(context_length + 1)
    indices = (start_indices + offsets).numpy()
    
    # 3. 一次性从内存抓取所有数据 (dataset 必须是 np.ndarray)
    batch_data = dataset[indices] 
    
    # 4. 转为 Tensor
    batch_tensor = torch.from_numpy(batch_data.astype(np.int64))
    
    # 5. 切分 x, y
    x = batch_tensor[:, :-1]
    y = batch_tensor[:, 1:]

    # 6. 搬运到 GPU
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

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


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # compute log(sum(exp(x)))
    # find max value
    m = inputs.max(dim=-1, keepdim=True).values

    # all value increase m
    inputs_safe = inputs - m

    # compute log_sum_exp
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_safe), dim=-1)) + m.squeeze(-1)

    # get target
    batch_indices = torch.arange(inputs.shape[0],device=inputs.device)
    true_logits = inputs[batch_indices, targets]

    # comnpute loss
    loss = log_sum_exp - true_logits

    return loss.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    params = [p for  p in parameters if p.grad is not None]
    
    if len(params) == 0:
        return
    
    # comput grads square sum
    device = params[0].grad.device
    total_norm_sq = torch.zeros([],device=device)

    for p in params:
        total_norm_sq += p.grad.detach().pow(2).sum()

    # total num
    total_norm = total_norm_sq.sqrt()

    # clipping
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)

        for p in params:
            p.grad.detach().mul_(scale)



def get_adamw_cls() -> AdamW:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # warm up progress
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # post_annealing progress 
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # Cosine annealing 
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    # cosine format
    cos_term = math.cos(progress * math.pi)

    # min + 0.5 * (1 + cos_term) * (max - min)
    return min_learning_rate + 0.5 * (1 + cos_term) * (max_learning_rate - min_learning_rate)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    # dict
    checkpoint_state = {
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "iteration" : iteration
    }
    
    # torch save
    torch.save(checkpoint_state, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    # load checkpoint
    checkpoint_state = torch.load(src, map_location = "cpu")
    
    # recover param
    model.load_state_dict(checkpoint_state["model_state_dict"])

    # recover op state
    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    # return step
    return checkpoint_state["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )

def merge_pair(
    word: tuple[bytes, ...], 
    pair: tuple[bytes, bytes], 
    new_token: bytes
) -> tuple[bytes, ...]:
    new_word = []
    i =0
    while i < len(word):
        # check if merge pair
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            new_word.append(new_token)
            # skip two addresses
            i += 2
        else:
            new_word.append(word[i])
            # skip one address
            i += 1
    return tuple(new_word)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from collections import Counter, defaultdict
    import regex as re
    import heapq

    # Read file
    with open(input_path, 'r', encoding='utf-8')as f:
        text = f.read()
    
    # Initial vocab
    # 256 bytes
    vocab  = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # add special_tokens
    for special_token in special_tokens:
        if special_token.encode('utf-8') not in vocab.values():
            vocab[len(vocab)] = special_token.encode('utf-8')

    # pretokenize
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # split text by special tokens
    if special_tokens:
        special_pattern = '|'.join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        text_parts = re.split(f'({special_pattern})', text)
    else:
        text_parts = [text]
    
    # get word counts
    word_counts = Counter()
    for part in text_parts:
        if part in special_tokens:
            word_counts[tuple([part.encode('utf-8')])] += 1
        elif part:
            tokens = re.findall(pat, part)
            for token in tokens:
                token_bytes = token.encode('utf-8')
                word_counts[tuple(bytes([b]) for b in token_bytes)] += 1

    # BPE training
    merges = []

    # set of special token bytes
    special_token_bytes = {st.encode('utf-8') for st in special_tokens}

    while len(vocab) < vocab_size:
        # Count all pairs
        pair_counts = Counter()
        
        for word, count in word_counts.items():
            # Skip special tokens (single-element tuples containing special token bytes)
            if len(word) == 1 and word[0] in special_token_bytes:
                continue
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count
        
        if not pair_counts:
            break
        
        # Find best pair: max count, with lexicographic tiebreaker
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # Merge new best pair and add to vocab
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)
        
        # Update word_counts
        new_word_counts = Counter()
        
        for word, count in word_counts.items():
            # Skip special tokens
            if len(word) == 1 and word[0] in special_token_bytes:
                new_word_counts[word] += count
                continue
            
            # Check if word contains best_pair
            has_best_pair = False
            for i in range(len(word) - 1):
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    has_best_pair = True
                    break
            
            if has_best_pair:
                # Merge the pair in this word
                new_word = merge_pair(word, best_pair, new_token)
                new_word_counts[new_word] += count
            else:
                new_word_counts[word] += count
        
        word_counts = new_word_counts

    return vocab, merges

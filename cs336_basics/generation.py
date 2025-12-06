import torch
import torch.nn.functional as F

def sample_next_token(logits: torch.Tensor, temperature: float=1.0, top_p: float = 0.0) -> torch.Tensor:
    # temperature scaling
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature

    # Top_p sampling
    if top_p > 0.0 and top_p < 1.0:
        # sort
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # compute prob and add
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # produce mask
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # return mask
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        # set -inf
        logits[indices_to_remove] = -float('inf')

    # final sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token

def generate(model,indices:torch.Tensor, max_new_token: int, temperature: float=1.0, top_p: float=0.0, eos_token_id: int=None) ->torch.Tensor:
    indices = indices.to(model.device)

    for _ in range(max_new_token):
        context_len = getattr(model, "context_length", 1024)
        idx_cond = indices if indices.szie(1) <= context_len else indices[:, -context_len:]

        # Forward propagation
        with torch.inference_mode():
            logits = model(idx_cond)

        # get last step logits
        logits = logits[:, -1, :]

        # sample next token
        idx_next = sample_next_token(logits, temperature, top_p)

        # put after indices
        indices = torch.cat((indices, idx_next),dim=-1)

        # if batch_szie=1 and generate end can stop
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break

    return indices

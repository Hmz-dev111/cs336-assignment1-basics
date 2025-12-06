import os
import time
import math
import argparse
import numpy as np
import torch

from cs336_basics.transformer_lm import transformer_lm
from cs336_basics.AdamW import AdamW
from tests.adapters import run_get_lr_cosine_schedule, run_get_batch, run_cross_entropy,run_gradient_clipping,run_save_checkpoint

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # set data and location
    parser.add_argument("--data_path", type=str, required=True, help="Path to the tokenized .bin file")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save checkpoints")

    # model params
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers",type=int,default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    # 在 parse_args 里增加这一行
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # training params
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay",type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # system set
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=10, help="how often to print")
    parser.add_argument("--save_interval", type=int, default=1000, help="how often to save")

    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project="cs336-transformer", config=args)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)

    print(f"using device: {args.device}")

    # data loading
    train_data = np.fromfile(args.data_path, dtype=np.uint16)
    print(f"data loaded. Total token:{len(train_data)}")

    # initial model
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device
    )

    # initial optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9,0.95)
    )

    # training loop
    model.train()
    start_time = time.time()

    for iter_num in range(1, args.max_iters + 1):
        # get batch
        x, y = run_get_batch(train_data, args.batch_size,args.context_length, args.device)

        # learning rate schedule
        lr = run_get_lr_cosine_schedule(it=iter_num, max_learning_rate=args.learning_rate, min_learning_rate=args.min_lr, warmup_iters=args.warmup_iters, cosine_cycle_iters=args.max_iters)

        # update lr in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # forward
        logits = model(x)

        # compute loss
        loss = run_cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )

        # brackward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # grad clipping
        if args.grad_clip != 0.0:
            run_gradient_clipping(model.parameters(), args.grad_clip)

        # updata step
        optimizer.step()

        # print
        if iter_num % args.log_interval == 0:
            dt = time.time() - start_time
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2}ms, lr {lr:.6f}")
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "iter": iter_num,
            })
            start_time = time.time()

        # save checkpoint
        if iter_num % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{iter_num}.pt")
            print(f"Saving checkpoint to {ckpt_path}...")
            run_save_checkpoint(model, optimizer, iter_num, ckpt_path)

    print("Training Finished!")

if __name__ == "__main__":
    main()


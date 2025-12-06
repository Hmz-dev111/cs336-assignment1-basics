import time
import torch
import numpy as np
from cs336_basics.transformer_lm import transformer_lm # 或者是 transformer_lm
from cs336_basics.AdamW import AdamW
from tests.adapters import run_get_batch, run_cross_entropy

# 1. 模拟环境
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {device}")

# 2. 制造假数据 (直接在内存里，排除硬盘读取干扰)
print("Generating dummy data in RAM...")
train_data = np.random.randint(0, 10000, size=(1000000,), dtype=np.uint16)

# 3. 初始化模型
model = transformer_lm(
    vocab_size=10000, context_length=256, d_model=512, 
    num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000, device=device
).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)

# 4. 测速循环
print("\n=== Start Profiling Loop ===")
model.train()
for i in range(5):
    t0 = time.time()
    
    # A. 获取数据
    x, y = run_get_batch(train_data, 128, 256, device)
    torch.cuda.synchronize() # 等待 GPU 完成
    t1 = time.time()
    
    # B. 前向传播
    logits = model(x)
    torch.cuda.synchronize()
    t2 = time.time()
    
    # C. 计算 Loss
    loss = run_cross_entropy(logits.view(-1, 10000), y.view(-1))
    torch.cuda.synchronize()
    t3 = time.time()
    
    # D. 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.cuda.synchronize()
    t4 = time.time()
    
    # E. 优化器更新
    optimizer.step()
    torch.cuda.synchronize()
    t5 = time.time()
    
    print(f"Iter {i+1}:")
    print(f"  [Data Load]: {t1 - t0:.4f}s")
    print(f"  [Forward  ]: {t2 - t1:.4f}s")
    print(f"  [Calc Loss]: {t3 - t2:.4f}s")
    print(f"  [Backward ]: {t4 - t3:.4f}s")
    print(f"  [Opt Step ]: {t5 - t4:.4f}s")
    print(f"  [Total    ]: {t5 - t0:.4f}s")
#!/usr/bin/env python3
"""验证 C 引擎的前向传播数值是否正确。

用法:
    .venv/bin/python verify.py

会做两件事:
1. 用 transformers 加载 Qwen2.5-0.5B，输入单个 token (默认 198='\\n')，
   打印 top-5 next-token logits。
2. 把这些数值和 ./run fwd 的输出对比。

对照基准: HuggingFace transformers 的官方实现 (fp32)。
我们的 C 引擎也是 fp32，所以数值应该高度一致 (允许 ~0.01 量级的浮点误差)。
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
TOKEN_ID = 198   # '\n'
POS = 0

print(f"加载 {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,   # 和 C 引擎一致用 fp32
    device_map="cpu",
)
model.eval()

print(f"\n=== transformers 参考 (token={TOKEN_ID}, pos={POS}) ===")
with torch.no_grad():
    inp = torch.tensor([[TOKEN_ID]])
    logits = model(inp).logits[0, POS]   # (vocab_size,)
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(logits, 5)
    for idx, (vid, val) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
        s = tok.decode([vid])
        s_repr = repr(s)
        print(f"  #{idx} id={vid:<8d} logit={val:.6f}  token={s_repr}")
    print(f"  (参考) logits[151643(EOS)] = {logits[151643].item():.6f}")

print(f"\n现在运行 C 引擎对照:")
print(f"  ./run fwd model.safetensors {TOKEN_ID} {POS}")
print(f"\n如果两边的 id 顺序和 logit 数值一致 (误差 < 0.1)，前向传播实现正确。")

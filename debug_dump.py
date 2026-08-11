#!/usr/bin/env python3
"""逐层 dump Qwen2.5-0.5B 的中间张量，和 C 引擎对照调试。

dump 内容 (单 token 输入, token_id=198, pos=0):
  - embedding 后的 x[0:8]
  - 第 0 层 rmsnorm 后的 xb[0:8]
  - 第 0 层 q_proj 后的 q[0:8]
  - 第 0 层 k_proj 后的 k[0:8]
  - 第 0 层 rope 后的 q[0:8]
  - 第 0 层 attention 输出后(attn out, 即 Wo 输入)的 xb[0:8]
  - 第 0 层 MLP 后的 x[0:8]
  - 最终 rmsnorm 后的 x[0:8]
  - logits 的 top-5

数值存成 JSON, 给 C 端对照。
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
TOKEN_ID = 198

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="cpu",
)
model.eval()

# hook 容器
dumps = {}

def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        dumps[name] = out.detach().float().cpu().numpy().flatten()
    return hook

# 注册 hook 到关键模块
m = model.model
# embedding 输出
m.embed_tokens.register_forward_hook(make_hook("embed"))
# 第 0 层
l0 = m.layers[0]
l0.input_layernorm.register_forward_hook(make_hook("L0_rmsnorm_in"))   # 这层输出=norm后
l0.self_attn.q_proj.register_forward_hook(make_hook("L0_q_proj"))
l0.self_attn.k_proj.register_forward_hook(make_hook("L0_k_proj"))
l0.self_attn.v_proj.register_forward_hook(make_hook("L0_v_proj"))
l0.self_attn.o_proj.register_forward_hook(make_hook("L0_o_proj_in"))   # 输入是 attn out
l0.post_attention_layernorm.register_forward_hook(make_hook("L0_rmsnorm_ffn"))
l0.mlp.down_proj.register_forward_hook(make_hook("L0_mlp_down_in"))
# 第 0 层完整输出
l0.register_forward_hook(make_hook("L0_out"))
# 最终 norm
m.norm.register_forward_hook(make_hook("final_norm"))

with torch.no_grad():
    inp = torch.tensor([[TOKEN_ID]])
    logits = model(inp).logits[0, -1]

result = {}
for k in ["embed", "L0_rmsnorm_in", "L0_q_proj", "L0_k_proj", "L0_o_proj_in", "L0_out", "final_norm"]:
    arr = dumps[k]
    result[k] = [float(x) for x in arr[:8]]
    print(f"{k}[0:8] = {result[k]}")

# top-5 logits
top = torch.topk(logits, 5)
result["top5"] = [(int(i), float(v)) for i, v in zip(top.indices, top.values)]
print(f"\ntop5 = {result['top5']}")
print(f"logits[151643] = {logits[151643].item()}")

with open("debug_ref.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n已写 debug_ref.json")

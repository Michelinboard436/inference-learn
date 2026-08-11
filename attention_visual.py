"""用 "cat dog" 实际演示 Attention 的每一步。

为什么用英文? 因为两个独立的英文词会被切成两个 token,
能清楚展示"token 之间的注意力"。
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B', torch_dtype=torch.float32, device_map='cpu'
)
model.eval()

text = "cat dog"
ids = tok(text, return_tensors='pt')['input_ids']
print(f'输入: {text}')
print(f'token ids: {ids.tolist()[0]}')
print(f'每个 token: {[tok.decode([i]) for i in ids[0]]}')
print(f'(共 {ids.shape[1]} 个 token)')
print()

with torch.no_grad():
    # ---- Embedding ----
    embed = model.model.embed_tokens(ids)  # (1, 2, 896)
    print('=' * 70)
    print('第 1 步: Embedding - token id 变成 896 维向量')
    print('=' * 70)
    print(f"'cat' 向量前5维: {[round(x,4) for x in embed[0,0,:5].tolist()]}")
    print(f"'dog' 向量前5维: {[round(x,4) for x in embed[0,1,:5].tolist()]}")
    print()

    # ---- RMSNorm ----
    x = model.model.layers[0].input_layernorm(embed)

    # ---- QKV 投影 ----
    attn = model.model.layers[0].self_attn
    q = attn.q_proj(x)  # (1, 2, 896)
    k = attn.k_proj(x)  # (1, 2, 128)
    v = attn.v_proj(x)  # (1, 2, 128)

    print('=' * 70)
    print('第 2 步: QKV 投影 - 同一个向量, 投影出 3 个不同视角')
    print('=' * 70)
    print()
    print('cat 的输入向量, 经过 3 个不同的权重矩阵, 得到 3 个不同向量:')
    print(f'  Q(cat) 前5维: {[round(x,4) for x in q[0,0,:5].tolist()]}  <- cat 想找什么')
    print(f'  K(cat) 前5维: {[round(x,4) for x in k[0,0,:5].tolist()]}  <- cat 是什么标签')
    print(f'  V(cat) 前5维: {[round(x,4) for x in v[0,0,:5].tolist()]}  <- cat 的内容')
    print()
    print('dog 同理:')
    print(f'  Q(dog) 前5维: {[round(x,4) for x in q[0,1,:5].tolist()]}')
    print(f'  K(dog) 前5维: {[round(x,4) for x in k[0,1,:5].tolist()]}')
    print(f'  V(dog) 前5维: {[round(x,4) for x in v[0,1,:5].tolist()]}')
    print()

    # ---- 切多头 (只看 head 0) ----
    q = q.view(1, 2, 14, 64)  # 14 个 Q 头
    k = k.view(1, 2, 2, 64)   # 2 个 KV 头
    v = v.view(1, 2, 2, 64)

    print('=' * 70)
    print('第 3 步: 切多头 (只演示 head 0)')
    print('=' * 70)
    print('Q 切成 14 个头(每头64维), K/V 切成 2 个头(每头64维)')
    print('head 0 的 Q 共用 KV head 0 的数据 (GQA: 7个Q头共享1个KV头)')
    print()

    # ---- 算注意力分数 ----
    q_cat = q[0, 0, 0]   # cat 的 head0 Q
    q_dog = q[0, 1, 0]   # dog 的 head0 Q
    k_cat = k[0, 0, 0]   # cat 的 head0 K
    k_dog = k[0, 1, 0]   # dog 的 head0 K

    import math
    scale = math.sqrt(64)

    print('=' * 70)
    print('第 4 步: Q . K 算分数 (点积 / 根号64)')
    print('=' * 70)
    print()
    print('cat (pos0) 拿自己的 Q 去和每个位置的 K 做点积:')
    s_cc = torch.dot(q_cat, k_cat).item() / scale
    print(f'  Q(cat) . K(cat) = {s_cc:+.3f}  <- cat 和自己的相似度')
    print(f'  (cat 看不到 dog, 因为是因果: 只能看左边)')
    print()

    print('dog (pos1) 拿自己的 Q 去和每个位置的 K 做点积:')
    s_dc = torch.dot(q_dog, k_cat).item() / scale
    s_dd = torch.dot(q_dog, k_dog).item() / scale
    print(f'  Q(dog) . K(cat) = {s_dc:+.3f}  <- dog 觉得 cat 跟自己有多相关')
    print(f'  Q(dog) . K(dog) = {s_dd:+.3f}  <- dog 觉得自己跟自己有多相关')
    print()

    # ---- softmax ----
    print('=' * 70)
    print('第 5 步: softmax - 分数变成概率(加起来=1)')
    print('=' * 70)
    print()

    att_cat = F.softmax(torch.tensor([s_cc]), dim=0)
    print(f'cat 的注意力: 只能看自己 -> [1.000] (没得选)')
    print()

    att_dog = F.softmax(torch.tensor([s_dc, s_dd]), dim=0)
    print(f'dog 的分数:   [{s_dc:+.3f}, {s_dd:+.3f}]')
    print(f'softmax 后:   [{att_dog[0]:.3f}, {att_dog[1]:.3f}]')
    print(f'解读: dog 把 {att_dog[0]*100:.0f}% 注意力放在 cat 上, {att_dog[1]*100:.0f}% 放在自己')
    print()

    # ---- 加权 V ----
    print('=' * 70)
    print('第 6 步: 用注意力权重加权 Value - 得到最终输出')
    print('=' * 70)
    print()
    v_cat = v[0, 0, 0]  # cat 的 head0 V
    v_dog = v[0, 1, 0]  # dog 的 head0 V

    out_cat = att_cat[0] * v_cat
    out_dog = att_dog[0] * v_cat + att_dog[1] * v_dog

    print(f'cat 的输出 = {att_cat[0]:.1f} x V(cat)')
    print(f'           = {[round(x,4) for x in out_cat[:4].tolist()]} (前4维)')
    print()
    print(f'dog 的输出 = {att_dog[0]:.3f} x V(cat) + {att_dog[1]:.3f} x V(dog)')
    print(f'           = {[round(x,4) for x in out_dog[:4].tolist()]} (前4维)')
    print()
    print('>>> 关键: dog 的输出里混入了 cat 的信息!')
    print('>>> 如果 dog 把 80% 注意力放在 cat 上,')
    print('>>> 那 dog 的输出就有 80% 是 cat 的内容')
    print('>>> 这就是注意力在做的事: 让每个词"看到"其他词')
    print()

    print('=' * 70)
    print('完整流程回顾')
    print('=' * 70)
    print('''
  cat(pos0)                          dog(pos1)
     |                                  |
  embed -> 896维向量                 embed -> 896维向量
     |                                  |
  投影出 Q,K,V                      投影出 Q,K,V
     |                                  |
  Q(cat).K(cat) 算分数              Q(dog).K(cat), Q(dog).K(dog) 算分数
  只能看自己(因果)                  能看 cat 和自己
     |                                  |
  softmax -> 100% 看自己            softmax -> X%看cat + Y%看自己
     |                                  |
  加权 V -> 输出                    加权 V -> 输出(混入了cat的信息)
     |                                  |
  ... 经过 24 层同样的处理 ...       ... 经过 24 层同样的处理 ...
     |                                  |
  最终向量 -> logits -> 预测下一个词
''')

# 第 9 章 调试与验证方法论

> 第 9 章 | [上一章：第 8 章 日志与可视化](./08-logging.md) | [下一章：第 10 章 生态对比](./10-ecosystem.md)

> 这章不是讲某个具体源码文件，而是讲**怎么证明你写的引擎是对的**。
> 作者认为这一章比代码本身更有学习价值：因为「能跑」和「算得对」之间隔着一整套方法论。

---

## 目录

- [为什么必须数值验证](#为什么必须数值验证)
- [单 token 验证法](#单-token-验证法)
- [ASan：编译器自带的内存错误检测器](#asan编译器自带的内存错误检测器)
- [逐层 dump：二分定位 bug](#逐层-dump二分定位-bug)
- [验证清单](#验证清单)
- [推荐学习路径](#推荐学习路径)
- [一句话总结](#一句话总结)

---

## 为什么必须数值验证

前向传播有 6 个算子（RMSNorm、matmul、RoPE、Attention、GQA、SwiGLU），**任何一个错了程序都能跑，不崩，但输出是垃圾**。

```
错的代码也会跑完, 也会吐出 200 个 "token"
  → 但内容是乱码 / 重复 / 不连贯
  → 你看到"模型在动", 却不知道哪里坏了
```

这不是推理引擎特有的问题，是所有**浮点数值计算**的通病：

| 错误类型 | 普通 CRUD 程序 | 数值程序 |
|---------|---------------|---------|
| 语法错 | 编译失败 | 编译失败 |
| 内存错 | 段错误崩溃 | **可能崩溃，也可能继续跑**（结果错） |
| 逻辑错 | 报错 / 结果明显不对 | **静默错**，输出看起来像那么回事 |

数值程序最可怕的就是「静默错」。你能拿到 logits，能采样出 token，能拼成文本：但每个数字都偏了一点点，累积 24 层后完全跑偏。

### 唯一可靠的验证：和官方实现逐数值对比

本项目用的对照基准是 **HuggingFace transformers 的官方 PyTorch 实现**（fp32）：

```
同样的输入 token + 同样的权重
  ├── PyTorch (transformers)  → top-5 logits = [id_a, id_b, id_c, id_d, id_e]
  └── 我们的 C 引擎           → top-5 logits = ?

如果两边 id 顺序一致 + 数值误差 < 0.1 → 前向传播实现正确
否则 → 哪里算错了, 需要定位
```

> 这是数值编程的黄金法则：**永远要有一个可信的 reference 实现**。在本项目里，PyTorch 就是那个 reference。

---

## 单 token 验证法

最有效的验证手段不是跑一整句生成（输出长、变量多、难定位），而是**输入单个 token，对比 top-5 logits**。

### 为什么单 token 就够

```
输入 "Hello world, today is"  (6 个 token)
  → 涉及 prefill + decode + KV Cache + 采样
  → 任何一步错都会让输出乱
  → 出了问题不知道怪哪一步

输入单个 token (如 198 = '\n', pos=0)
  → 只有一次 forward
  → 没有 KV Cache 累积误差
  → top-5 logits 一眼能对
  → 错了就一定是 forward 内部某个算子的问题
```

单 token 把变量降到最少：这是**控制变量法**在调试中的应用。

### 操作步骤

**第 1 步：C 引擎算**

```bash
./run fwd model.safetensors 198 0
```

`fwd` 是 `run.c` 的调试子命令：加载权重，输入一个 token id，打印 top-5 logits。

输出形如：

```
=== top-5 next-token (input token=198, pos=0) ===
  id=198      logit=11.842103
  id=618      logit=10.234567
  id=13       logit=9.876543
  ...
  (参考) logits[151643(EOS)] = -2.345678
```

**第 2 步：PyTorch 算**

```bash
.venv/bin/python verify.py
```

`verify.py` 用 transformers 加载同一个模型，输入同一个 token，打印同样的 top-5。

**第 3 步：人眼对照**

```
C 引擎                          PyTorch
─────────────────────────────────────────────
id=198  logit=11.842103         id=198  logit=11.842098
id=618  logit=10.234567         id=618  logit=10.234571
id=13   logit=9.876543          id=13   logit=9.876540
...                             ...

→ id 顺序一致 ✓
→ logit 数值误差 < 0.001 ✓
→ 前向传播实现正确!
```

### 容许的误差范围

两个 fp32 实现之间不可能完全一致（运算顺序不同、`expf` 实现不同、累加顺序不同），但误差应该很小：

| 误差 | 判断 |
|------|------|
| `< 0.01` | 完全正常，浮点噪声 |
| `0.01 - 0.1` | 可接受，可能只是累加顺序差异 |
| `0.1 - 1.0` | 可疑，需要逐层排查 |
| `> 1.0` | 一定有 bug |
| top-5 id 顺序变了 | 一定有 bug（即使数值接近） |

### 不同 token 多测几次

单个 token 对了不代表全对：可能刚好这个 token 的路径上没踩到 bug。多测几个：

```bash
./run fwd model.safetensors 198 0      # '\n'
./run fwd model.safetensors 9707 0     # 'Hello'
./run fwd model.safetensors 4616 0     # 'cat'
./run fwd model.safetensors 5562 0     # ' dog'
```

每个都和 PyTorch 对照，全部一致才算验证通过。

---

## ASan：编译器自带的内存错误检测器

光数值对还不够：可能存在**内存越界**但刚好没崩（写了别人的内存但没被发现的「定时炸弹」）。这类问题靠人眼审查找不出来，要用工具。

### AddressSanitizer 简介

```bash
gcc -fsanitize=address,undefined -o run_dbg *.c -lm
```

`-fsanitize=address` 开启 **AddressSanitizer（ASan）**，`-fsanitize=undefined` 开启 **UndefinedBehaviorSanitizer（UBSan）**。两者都是 GCC/Clang 自带的。

ASan 能抓的错：

| 错误类型 | 说明 |
|---------|------|
| 堆越界（heap-buffer-overflow） | `malloc(100)` 但访问了第 101 字节 |
| 栈越界（stack-buffer-overflow） | `int a[10]; a[15]=0;` |
| 全局变量越界 | 全局数组越界访问 |
| Use-after-free | `free(p); *p;` |
| Double-free | 同一块内存 free 两次 |
| 内存泄漏（leak） | malloc 了没 free |

UBSan 能抓的错：

| 错误类型 | 说明 |
|---------|------|
| 整数溢出（signed overflow） | `INT_MAX + 1` |
| 空指针解引用 | `*NULL` |
| 除以零 | `int x = 1/0;` |
| 类型违反 | 错误的类型转换 |

### 工作原理（粗略）

ASan 给每块分配的内存加上「红区」（redzone），访问红区就报错：

```
malloc(100) 实际布局:
  ┌──────────────────────────────────┐
  │ 红区  │  你能用的 100 字节  │ 红区  │
  └──────────────────────────────────┘
   ↑                          ↑
   访问这里就报错           访问这里也报错
```

代价是**内存占用增加 2-3 倍、速度变慢 2 倍左右**：所以只在调试构建用，正式构建关掉。

### 用法

```bash
# 编译带 sanitizer 的版本
gcc -fsanitize=address,undefined -O1 -g -o run_dbg *.c -lm

# 正常跑 (有内存错会立刻报错)
./run_dbg fwd model.safetensors 198 0
```

出错时输出形如：

```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000000d4
READ of size 4 at 0x6020000000d4 thread T0
    #0 0x10a2b3c40 in matmul net.c:118
    #1 0x10a2b4d20 in forward net.c:226
    #2 0x10a2b5e10 in main run.c:234
...
```

直接告诉你**哪个文件哪一行**越界了。

### 本项目的实战经验

> 作者按：本项目开发时，两个最严重的 bug 都是 ASan 抓到的：

**Bug 1：`matmul` 参数传反**

```c
// 错的写法:
matmul(out, W, x, bias, d, n);   // n 和 d 传反了!

// 正确:
matmul(out, W, x, bias, n, d);   // n=输入维, d=输出维
```

参数传反会导致 `W + i * n` 算出错误的行偏移，访问到不属于这块内存的地方。人眼看不出来（参数顺序很容易混淆），但 ASan 立刻报越界。

**Bug 2：KV Cache 索引错**

cache 的索引是 `layer * seq_len * kv_dim + pos * kv_dim + head * head_dim`，任何一项乘错都会越界。这类 bug 光靠数值对不上很难定位（输出看着像那么回事），ASan 直接指出哪一行越界。

### 调试构建 vs 正式构建

```makefile
# 正式构建 (Makefile 默认)
gcc -O2 -std=c99 -o run *.c -lm

# 调试构建 (手动)
gcc -fsanitize=address,undefined -O1 -g -o run_dbg *.c -lm
```

| | 正式 (`run`) | 调试 (`run_dbg`) |
|---|---|---|
| 速度 | 快 | 慢 2 倍 |
| 内存 | 正常 | 多 2-3 倍 |
| 抓内存错 | 不抓（可能崩溃可能不崩） | 立刻精确报错 |
| 用途 | 日常运行、性能测试 | 开发、验证、找 bug |

> 建议：每次写新算子或改维度相关代码后，先用 `run_dbg` 跑一遍单 token，确认无内存错，再用 `run` 测数值。

---

## 逐层 dump：二分定位 bug

单 token top-5 对不上时，问题可能在前向传播的任何一步。怎么缩小范围？**逐层 dump 中间张量**。

### 思路：在 C 和 PyTorch 两边 dump 同一位置的中间值，逐步对比

```
forward() 有 24 层, 每层有 ~10 个中间张量
  → 全 dump 太多, 没法看

二分思路:
  ① 先 dump 第 0 层的输出
     对得上?  → bug 在第 1-23 层
     对不上?  → bug 在第 0 层内部, 继续细分
  ② 第 0 层内部再细分: rmsnorm 后对不对? q_proj 后对不对? ...
  ③ 第一个对不上的地方 = bug 所在
```

### 工具：debug_dump.py

本项目带的 `debug_dump.py` 用 PyTorch 的 hook 机制，dump 第 0 层的关键中间张量：

```python
def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        dumps[name] = out.detach().float().cpu().numpy().flatten()
    return hook

# 注册 hook 到关键模块
l0 = m.layers[0]
l0.input_layernorm.register_forward_hook(make_hook("L0_rmsnorm_in"))
l0.self_attn.q_proj.register_forward_hook(make_hook("L0_q_proj"))
l0.self_attn.k_proj.register_forward_hook(make_hook("L0_k_proj"))
l0.self_attn.o_proj.register_forward_hook(make_hook("L0_o_proj_in"))
l0.mlp.down_proj.register_forward_hook(make_hook("L0_mlp_down_in"))
l0.register_forward_hook(make_hook("L0_out"))
```

运行后输出 JSON：

```json
{
  "embed":         [-0.0162, 0.0031, -0.0019, ...],
  "L0_rmsnorm_in": [...],
  "L0_q_proj":     [...],
  "L0_k_proj":     [...],
  "L0_o_proj_in":  [...],
  "L0_out":        [...],
  "final_norm":    [...]
}
```

### C 端的对应物

C 引擎开 trace 级（`-vv`）会打印同样的中间张量：

```bash
./run fwd model.safetensors 198 0 dump
#                                                     ↑
#                                  第 5 个参数 "dump" 打开逐层 dump
```

`run.c` 里：

```c
if (argc >= 6 && strcmp(argv[5], "dump") == 0) {
    net_debug_dump = 1;   /* 全局开关, net.c 里 forward 会打印中间张量 */
}
```

`net.c` 的 forward 里每个算子后面判断这个开关，打印前 8 个数。

### 对照流程

```
PyTorch (debug_dump.py)        C 引擎 (run fwd ... dump)
─────────────────────────────────────────────────────────────
embed[0:8]    = [-0.016, ...]   embed[0:8]    = [-0.016, ...]   ✓
L0_rmsnorm[0:8] = [...]         L0_rmsnorm[0:8] = [...]        ✓
L0_q_proj[0:8] = [...]          L0_q_proj[0:8] = [...]         ✓
L0_k_proj[0:8] = [...]          L0_k_proj[0:8] = [...]         ✗ ← 这里开始不一致!
L0_o_proj_in   = [...]          L0_o_proj_in   = [...]
...
```

**第一个对不上的地方就是 bug 所在**：在上面的例子里，K 投影之后数值就错了，所以 bug 一定在 K 投影或它之前的 RoPE。

> 这就是为什么第 8 章的日志系统要支持 trace 级打印前几个数：它是这个对照流程的基础设施。

### 常见的「对不上」原因

逐层排查时，常见的 bug 类型：

| 现象 | 可能原因 |
|------|---------|
| 第 0 层 embed 就对不上 | embedding 索引错 / bf16→fp32 转换错 |
| rmsnorm 后对不上 | eps 位置错（先加还是后加）/ 没开根号 / weight 加反 |
| q_proj 对不上 | matmul 参数 n/d 传反 / bias 加错 / weight 加载错位 |
| RoPE 后对不上 | 角度算错 / 配对方式错（HF half 模式 vs 普通） |
| attention 后对不上 | 没除 √d / 没做 causal / softmax 数值不稳 |
| 第 0 层对但第 5 层不对 | 残差连接错 / weight 跨层索引错 |

每种 bug 都有自己的「指纹」：对不上的位置和方式能直接指向问题。

---

## 验证清单

把上面的方法整理成一个可勾选的清单。每次改了前向传播相关代码，建议过一遍：

### 数值验证

- [ ] 用 `run fwd` 跑单 token（至少 3 个不同的 token id）
- [ ] 用 `verify.py` 跑 PyTorch 对照
- [ ] top-5 id 顺序一致
- [ ] logit 数值误差 < 0.1
- [ ] EOS logits 数值也对得上

### 内存验证

- [ ] 用 `run_dbg`（ASan 版）跑一遍，无报错
- [ ] 改了维度相关代码后，重新跑 ASan
- [ ] 无内存泄漏（ASan 退出时报 `LEAK`）

### 中间张量验证（出问题时）

- [ ] 用 `run fwd ... dump` dump C 端中间张量
- [ ] 用 `debug_dump.py` dump PyTorch 端
- [ ] 逐层对照，找到第一个对不上的位置
- [ ] 在那个位置细分算子，定位 bug

### 生成质量验证

- [ ] `-t 0` 贪心生成，输出连贯且可复现
- [ ] 多个 prompt 都能生成合理文本
- [ ] 遇到 EOS 能正确停止
- [ ] 生成长度合理（不无限循环、不提前截断）

### 日志健康检查

- [ ] `-v` 看每层范数，应该稳定（不爆炸不消失）
- [ ] MLP 激活率合理（0.3-0.7，不全开不全关）
- [ ] 注意力分布有区分度（不是均匀的 1/n）

### 跨平台（如需）

- [ ] 同样的 seed 在不同平台生成同样的输出（验证 PRNG 一致性）
- [ ] 同样的权重在不同机器数值一致（验证 bf16 转换）

---

## 推荐学习路径

> 本节取材自 BOOK.md 附录 C，给出读完前 9 章后的建议路径。

### 第一周：跑起来 + 建立直觉

1. 读 README，下载权重，`make` 编译运行
2. 读第 1 章（基础概念）
3. 读第 4 章（`net.h` 三大结构体）
4. 跑三个子命令感受一下：
   ```bash
   ./run info  model.safetensors          # 看张量列表
   ./run encode tokenizer.bin "Hello"     # 看 BPE 编码
   ./run fwd  model.safetensors 198 0     # 看单 token logits
   ```

### 第二周：理解前向传播

1. 读第 5 章（`net.c`），重点看 `forward()` 函数
2. 每个算子对照「原理」部分
3. 跑 `./run fwd ... -v`，对照日志看每步中间值
4. 不懂的术语查附录 A 或 `net-h-glossary.md`

### 第三周：理解周边系统

1. 读第 2 章（safetensors 加载）
2. 读第 3 章（JSON 解析器）
3. 读第 6 章（分词器）
4. 读第 7 章（采样）

### 第四周：验证 + 拓展视野

1. 读本章（第 9 章），跑 `verify.py` 对比数值
2. 用 `run_dbg`（ASan 版）验证无内存错
3. 读第 10 章（生态对比），理解工业引擎
4. 尝试加一个优化（如 SIMD 加速 matmul）

### 验证里程碑

每周末用一个简单的验证确认这周学懂了：

| 周次 | 验证 |
|------|------|
| 第 1 周末 | `./run info` 能跑，看到 embedding 表的 shape |
| 第 2 周末 | 能解释 `forward()` 每一步在做什么 |
| 第 3 周末 | 能解释 BPE 怎么切词、采样怎么选 token |
| 第 4 周末 | C 引擎和 PyTorch 的 top-5 logits 对得上 |

---

## 一句话总结

**数值验证 = 单 token 控制变量（top-5 对答案）+ ASan 抓内存错 + 逐层 dump 二分定位**。
这套方法论适用于任何数值计算项目，不只是推理引擎。
本项目能跑出和 PyTorch 一致的输出，靠的不是「写对了」，而是「错了能用这套方法找出来再改对」。

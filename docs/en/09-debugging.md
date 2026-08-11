# Chapter 9 · Debugging & Verification Methodology

> 🌐 [中文版](../cn/09-debugging) | English

> Chapter 9 | [Previous: Chapter 8 · Logging & Visualization](./08-logging.md) | [Next: Chapter 10 · Ecosystem Comparison](./10-ecosystem.md)

> This chapter isn't about a specific source file — it's about **how to prove the engine you wrote is correct**.
> The author considers this chapter more valuable than the code itself, because the gap between "it runs" and "it computes correctly" is bridged by a whole methodology.

---

## Table of Contents

- [Why Numerical Verification Is Essential](#why-numerical-verification-is-essential)
- [The Single-Token Verification Method](#the-single-token-verification-method)
- [ASan: The Compiler's Built-in Memory Error Detector](#asan-the-compilers-built-in-memory-error-detector)
- [Layer-by-Layer Dump: Bisecting to Locate Bugs](#layer-by-layer-dump-bisecting-to-locate-bugs)
- [Verification Checklist](#verification-checklist)
- [Recommended Learning Path](#recommended-learning-path)
- [One-Sentence Summary](#one-sentence-summary)

---

## Why Numerical Verification Is Essential

The forward pass has 6 operators (RMSNorm, matmul, RoPE, Attention, GQA, SwiGLU). **Get any one of them wrong and the program still runs — it doesn't crash, but the output is garbage**.

```
Incorrect code also runs to completion and spits out 200 "tokens"
  → but the content is gibberish / repetitive / incoherent
  → you see "the model is doing something", but you don't know what's broken
```

This isn't a problem unique to inference engines — it's a common affliction of all **floating-point numerical computation**:

| Error type | A typical CRUD program | A numerical program |
|---------|---------------|---------|
| Syntax error | Compile failure | Compile failure |
| Memory error | Segfault crash | **May crash, may also keep running** (with wrong results) |
| Logic error | Throws / obviously wrong result | **Silent error**, the output looks plausible |

The most terrifying thing about numerical programs is the "silent error." You can get logits, you can sample tokens, you can stitch them into text — but every number is slightly off, and after accumulating across 24 layers everything goes completely off the rails.

### The Only Reliable Verification: Compare Every Number Against the Official Implementation

The reference baseline used by this project is **HuggingFace transformers' official PyTorch implementation** (fp32):

```
Same input tokens + same weights
  ├── PyTorch (transformers)  → top-5 logits = [id_a, id_b, id_c, id_d, id_e]
  └── Our C engine             → top-5 logits = ?

If both sides have the same id order + numerical error < 0.1 → forward pass is correct
Otherwise → something is computed wrong, need to locate it
```

> This is the golden rule of numerical programming: **always have a trustworthy reference implementation**. In this project, PyTorch is that reference.

---

## The Single-Token Verification Method

The most effective verification technique isn't running a full sentence generation (long output, many variables, hard to locate issues), but **feeding a single token and comparing the top-5 logits**.

### Why a Single Token Is Enough

```
Input "Hello world, today is"  (6 tokens)
  → involves prefill + decode + KV Cache + sampling
  → any single step going wrong makes the output garbage
  → when something goes wrong, you don't know which step to blame

Input a single token (e.g. 198 = '\n', pos=0)
  → only one forward pass
  → no KV Cache accumulation error
  → top-5 logits are easy to compare at a glance
  → if it's wrong, it must be some operator inside forward
```

A single token minimizes the variables — this is the **method of controlled variables** applied to debugging.

### Procedure

**Step 1: Run the C engine**

```bash
./run fwd model.safetensors 198 0
```

`fwd` is `run.c`'s debug subcommand: it loads the weights, takes one token id as input, and prints the top-5 logits.

Output looks like:

```
=== top-5 next-token (input token=198, pos=0) ===
  id=198      logit=11.842103
  id=618      logit=10.234567
  id=13       logit=9.876543
  ...
  (reference) logits[151643(EOS)] = -2.345678
```

**Step 2: Run PyTorch**

```bash
.venv/bin/python verify.py
```

`verify.py` uses transformers to load the same model, feed the same token, and print the same top-5.

**Step 3: Compare by eye**

```
C engine                         PyTorch
─────────────────────────────────────────────
id=198  logit=11.842103         id=198  logit=11.842098
id=618  logit=10.234567         id=618  logit=10.234571
id=13   logit=9.876543          id=13   logit=9.876540
...                             ...

→ id order matches ✓
→ logit numerical error < 0.001 ✓
→ forward pass is correct!
```

### Tolerable Error Range

Two fp32 implementations can never be perfectly identical (different operation orders, different `expf` implementations, different accumulation orders), but the error should be small:

| Error | Verdict |
|------|------|
| `< 0.01` | Completely normal, floating-point noise |
| `0.01 - 0.1` | Acceptable, likely just accumulation-order differences |
| `0.1 - 1.0` | Suspicious, needs layer-by-layer investigation |
| `> 1.0` | Definitely a bug |
| top-5 id order changed | Definitely a bug (even if numbers are close) |

### Test Multiple Different Tokens

A single token being right doesn't mean everything is right — the bug may simply not lie on that token's path. Test a few more:

```bash
./run fwd model.safetensors 198 0      # '\n'
./run fwd model.safetensors 9707 0     # 'Hello'
./run fwd model.safetensors 4616 0     # 'cat'
./run fwd model.safetensors 5562 0     # ' dog'
```

Compare each against PyTorch. Only when all of them match is the verification considered passed.

---

## ASan: The Compiler's Built-in Memory Error Detector

Just getting the numbers right isn't enough — there may be **out-of-bounds memory access** that happens not to crash (writing into someone else's memory without being noticed, a "time bomb"). Such problems can't be found by eye review; you need tools.

### AddressSanitizer Overview

```bash
gcc -fsanitize=address,undefined -o run_dbg *.c -lm
```

`-fsanitize=address` enables **AddressSanitizer (ASan)**, `-fsanitize=undefined` enables **UndefinedBehaviorSanitizer (UBSan)**. Both are built into GCC/Clang.

**What ASan catches**:

| Error type | Description |
|---------|------|
| Heap buffer overflow | `malloc(100)` but accessing the 101st byte |
| Stack buffer overflow | `int a[10]; a[15]=0;` |
| Global variable overflow | Out-of-bounds access on a global array |
| Use-after-free | `free(p); *p;` |
| Double-free | Freeing the same block twice |
| Memory leak | `malloc` without `free` |

**What UBSan catches**:

| Error type | Description |
|---------|------|
| Signed integer overflow | `INT_MAX + 1` |
| Null pointer dereference | `*NULL` |
| Division by zero | `int x = 1/0;` |
| Type violation | Erroneous type cast |

### How It Works (Roughly)

ASan adds a "redzone" around every allocated block; accessing the redzone triggers an error:

```
malloc(100) actual layout:
  ┌──────────────────────────────────┐
  │ redzone │  your 100 usable bytes  │ redzone │
  └──────────────────────────────────┘
   ↑                          ↑
   accessing here is an error   accessing here is also an error
```

The cost is **2-3x higher memory usage and ~2x slower speed** — which is why it's only used in debug builds and turned off in production builds.

### Usage

```bash
# Build the sanitized version
gcc -fsanitize=address,undefined -O1 -g -o run_dbg *.c -lm

# Run normally (any memory error will be reported immediately)
./run_dbg fwd model.safetensors 198 0
```

On error the output looks like:

```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000000d4
READ of size 4 at 0x6020000000d4 thread T0
    #0 0x10a2b3c40 in matmul net.c:118
    #1 0x10a2b4d20 in forward net.c:226
    #2 0x10a2b5e10 in main run.c:234
...
```

It tells you directly **which file and which line** went out of bounds.

### Lessons Learned in This Project

> Author's note: during this project's development, **the two most severe bugs were both caught by ASan**:

**Bug 1: `matmul` arguments reversed**

```c
// wrong:
matmul(out, W, x, bias, d, n);   // n and d swapped!

// correct:
matmul(out, W, x, bias, n, d);   // n=input dim, d=output dim
```

Reversing the arguments causes `W + i * n` to compute the wrong row offset, accessing memory that doesn't belong to this block. You can't tell by eye (argument order is easy to mix up), but ASan immediately reports the overflow.

**Bug 2: KV Cache index wrong**

The cache index is `layer * seq_len * kv_dim + pos * kv_dim + head * head_dim`. Getting any one term's multiplier wrong causes an overflow. Such bugs are hard to locate just from mismatched numbers (the output looks plausible), but ASan directly points to the offending line.

### Debug Build vs Production Build

```makefile
# Production build (Makefile default)
gcc -O2 -std=c99 -o run *.c -lm

# Debug build (manual)
gcc -fsanitize=address,undefined -O1 -g -o run_dbg *.c -lm
```

| | Production (`run`) | Debug (`run_dbg`) |
|---|---|---|
| Speed | Fast | 2x slower |
| Memory | Normal | 2-3x more |
| Catches memory errors | No (may or may not crash) | Immediate, precise report |
| Use case | Daily runs, performance tests | Development, verification, finding bugs |

> **Recommendation**: every time you write a new operator or change dimension-related code, first run a single token through `run_dbg` to confirm there are no memory errors, then use `run` to test the numbers.

---

## Layer-by-Layer Dump: Bisecting to Locate Bugs

When the single-token top-5 doesn't match, the problem could be in any step of the forward pass. How do you narrow it down? **Dump intermediate tensors layer by layer**.

### Idea: dump the intermediate values at the same location on both the C and PyTorch sides, and compare step by step

```
forward() has 24 layers, each with ~10 intermediate tensors
  → dumping all of them is too much to read

Bisection approach:
  1. First dump layer 0's output
     matches?      → bug is in layers 1-23
     doesn't match? → bug is inside layer 0, subdivide further
  2. Subdivide inside layer 0: does rmsnorm output match? q_proj output? ...
  3. The first mismatched location = where the bug is
```

### Tool: debug_dump.py

The `debug_dump.py` included in this project uses PyTorch's hook mechanism to dump layer 0's key intermediate tensors:

```python
def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        dumps[name] = out.detach().float().cpu().numpy().flatten()
    return hook

# Register hooks on key modules
l0 = m.layers[0]
l0.input_layernorm.register_forward_hook(make_hook("L0_rmsnorm_in"))
l0.self_attn.q_proj.register_forward_hook(make_hook("L0_q_proj"))
l0.self_attn.k_proj.register_forward_hook(make_hook("L0_k_proj"))
l0.self_attn.o_proj.register_forward_hook(make_hook("L0_o_proj_in"))
l0.mlp.down_proj.register_forward_hook(make_hook("L0_mlp_down_in"))
l0.register_forward_hook(make_hook("L0_out"))
```

Running it outputs JSON:

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

### The C-Side Counterpart

The C engine at trace level (`-vv`) prints the same intermediate tensors:

```bash
./run fwd model.safetensors 198 0 dump
#                                                     ↑
#                          the 5th argument "dump" enables layer-by-layer dump
```

In `run.c`:

```c
if (argc >= 6 && strcmp(argv[5], "dump") == 0) {
    net_debug_dump = 1;   /* global switch, forward in net.c will print intermediate tensors */
}
```

In `net.c`'s forward, after each operator this switch is checked and the first 8 numbers are printed.

### Comparison Workflow

```
PyTorch (debug_dump.py)        C engine (run fwd ... dump)
─────────────────────────────────────────────────────────────
embed[0:8]    = [-0.016, ...]   embed[0:8]    = [-0.016, ...]   ✓
L0_rmsnorm[0:8] = [...]         L0_rmsnorm[0:8] = [...]        ✓
L0_q_proj[0:8] = [...]          L0_q_proj[0:8] = [...]         ✓
L0_k_proj[0:8] = [...]          L0_k_proj[0:8] = [...]         ✗ ← diverges starting here!
L0_o_proj_in   = [...]          L0_o_proj_in   = [...]
...
```

**The first mismatched location is where the bug is** — in the example above, the numbers go wrong right after the K projection, so the bug must be in the K projection or in the RoPE just before it.

> This is exactly why the logging system in Chapter 8 has to support printing the first few numbers at trace level — it's the infrastructure that underpins this comparison workflow.

### Common Causes of "Mismatch"

When investigating layer by layer, common bug types:

| Symptom | Possible cause |
|------|---------|
| Mismatch starts right at embed | Wrong embedding index / wrong bf16→fp32 conversion |
| Mismatch after rmsnorm | Wrong eps placement (added before vs after) / missing square root / weight added backwards |
| q_proj mismatch | matmul n/d arguments reversed / wrong bias / weight loaded with an offset |
| Mismatch after RoPE | Wrong angle / wrong pairing (HF half mode vs normal) |
| Mismatch after attention | Didn't divide by √d / no causal mask / softmax numerically unstable |
| Layer 0 matches but layer 5 doesn't | Wrong residual connection / wrong cross-layer weight indexing |

Each kind of bug has its own "fingerprint" — the position and manner of the mismatch directly point to the problem.

---

## Verification Checklist

Organize the methods above into a checkable list. **After every change to forward-pass-related code, it's recommended to go through this**:

### Numerical Verification

- [ ] Run a single token with `run fwd` (at least 3 different token ids)
- [ ] Run the PyTorch comparison with `verify.py`
- [ ] top-5 id order matches
- [ ] logit numerical error < 0.1
- [ ] EOS logit value also matches

### Memory Verification

- [ ] Run once with `run_dbg` (ASan version), no errors
- [ ] After changing dimension-related code, re-run ASan
- [ ] No memory leaks (ASan reports `LEAK` on exit)

### Intermediate Tensor Verification (when something goes wrong)

- [ ] Dump C-side intermediate tensors with `run fwd ... dump`
- [ ] Dump PyTorch-side with `debug_dump.py`
- [ ] Compare layer by layer, find the first mismatched position
- [ ] Subdivide the operators at that position, locate the bug

### Generation Quality Verification

- [ ] `-t 0` greedy generation, output is coherent and reproducible
- [ ] Multiple prompts all produce reasonable text
- [ ] Stops correctly when EOS is hit
- [ ] Generation length is reasonable (no infinite loop, no premature truncation)

### Log Health Check

- [ ] `-v` shows per-layer norms, should be stable (no explosion or vanishing)
- [ ] MLP activation ratio is reasonable (0.3-0.7, not all on or all off)
- [ ] Attention distribution is discriminating (not a uniform 1/n)

### Cross-Platform (if needed)

- [ ] Same seed produces the same output on different platforms (verifies PRNG consistency)
- [ ] Same weights give identical numbers on different machines (verifies bf16 conversion)

---

## Recommended Learning Path

> This section is adapted from BOOK.md Appendix C and gives a suggested path after finishing the first 9 chapters.

### Week 1: Get it running + build intuition

1. Read the README, download the weights, `make` and run
2. Read Chapter 1 (basic concepts)
3. Read Chapter 4 (the three core structs in `net.h`)
4. Run the three subcommands to get a feel:
   ```bash
   ./run info  model.safetensors          # view the tensor list
   ./run encode tokenizer.bin "Hello"     # view BPE encoding
   ./run fwd  model.safetensors 198 0     # view single-token logits
   ```

### Week 2: Understand the forward pass

1. Read Chapter 5 (`net.c`), focus on the `forward()` function
2. Compare each operator against the "principle" section
3. Run `./run fwd ... -v`, follow the logs to see each intermediate value
4. Look up unfamiliar terms in Appendix A or `net-h-glossary.md`

### Week 3: Understand the surrounding systems

1. Read Chapter 2 (safetensors loading)
2. Read Chapter 3 (JSON parser)
3. Read Chapter 6 (tokenizer)
4. Read Chapter 7 (sampling)

### Week 4: Verify + broaden your view

1. Read this chapter (Chapter 9), run `verify.py` to compare numbers
2. Use `run_dbg` (ASan version) to verify no memory errors
3. Read Chapter 10 (ecosystem comparison), understand industrial engines
4. Try adding an optimization (e.g. SIMD-accelerated matmul)

### Verification Milestones

At the end of each week, use a simple check to confirm you've learned that week's material:

| Week | Verification |
|------|------|
| End of week 1 | `./run info` runs, you see the embedding table's shape |
| End of week 2 | You can explain what each step of `forward()` does |
| End of week 3 | You can explain how BPE splits words and how sampling picks a token |
| End of week 4 | The C engine's top-5 logits match PyTorch's |

---

## One-Sentence Summary

**Numerical verification = single-token controlled variables (top-5 against the answer) + ASan for memory errors + layer-by-layer dump bisection**.
This methodology applies to any numerical computing project, not just inference engines.
This project can produce output identical to PyTorch not because it "was written correctly the first time", but because "when something is wrong, you can find it with these methods and then fix it".

---
layout: home

hero:
  name: "inference-learn"
  text: "Understand LLM Inference Engines from Scratch"
  tagline: Load Qwen2.5-0.5B with ~2700 lines of pure C. No PyTorch, just libc. Every component written from scratch.
  actions:
    - theme: brand
      text: 📖 Start Reading
      link: /en/01-basics
    - theme: alt
      text: 📋 Table of Contents
      link: /en/README
    - theme: alt
      text: 💻 Source Code
      link: https://github.com/jackiesre721/inference-learn

features:
  - icon: ⚡
    title: Pure C, Zero Dependencies
    details: No PyTorch, no AI frameworks, just libc. ~2700 lines of code, every operator hand-written and readable.
  - icon: 🧠
    title: Every Operator from Scratch
    details: RMSNorm, RoPE rotary positional encoding, GQA grouped attention, SwiGLU gated activation, KV Cache — each one broken down from math to code.
  - icon: 🔢
    title: Numerically Verified
    details: Forward pass results compared against the official PyTorch implementation. Error < 0.0002. Greedy generation matches HF token-by-token.
  - icon: 📖
    title: 10-Chapter Documentation
    details: From "what is an inference engine" to "how does it compare to vLLM" — 6700+ lines of docs with 13 Mermaid diagrams.
  - icon: 🔍
    title: Observable
    details: Tiered logging system + HTML visualization reports. Add -v for per-layer summaries, --report for a foldable execution trace.
  - icon: 🌍
    title: Industry Comparison
    details: Detailed comparison with llama.cpp / vLLM / SGLang. Understand what your engine is missing and why.
---

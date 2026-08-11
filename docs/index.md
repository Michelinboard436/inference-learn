---
layout: home

hero:
  name: "inference-learn"
  text: "从零理解 LLM 推理引擎"
  tagline: 用 ~2700 行纯 C 加载 Qwen2.5-0.5B，手写每一个零件。不依赖 PyTorch，只靠 libc。
  actions:
    - theme: brand
      text: 📖 开始阅读
      link: /01-basics
    - theme: alt
      text: 📋 目录
      link: /README
    - theme: alt
      text: 💻 源码仓库
      link: https://github.com/jackiesre721/inference-learn

features:
  - icon: ⚡
    title: 纯 C 实现，0 依赖
    details: 不用 PyTorch，不用任何 AI 框架，只靠 libc。~2700 行代码全部可读，每个算子都是手写的三重循环。
  - icon: 🧠
    title: 手写全部算子
    details: RMSNorm、RoPE 旋转位置编码、GQA 分组注意力、SwiGLU 门控激活、KV Cache —— 从公式到代码逐一拆解。
  - icon: 🔢
    title: 数值已验证
    details: 前向传播结果和 PyTorch 官方实现对比，误差 < 0.0002。贪心生成和 HF 逐 token 一致。
  - icon: 📖
    title: 10 章中文文档
    details: 从「什么是推理引擎」到「和 vLLM 差在哪」，6700 行文档 + 13 张 Mermaid 可视化图。
  - icon: 🔍
    title: 可观测
    details: 分级日志系统 + HTML 可视化报告。加 -v 看每层摘要，加 --report 生成可折叠的执行轨迹。
  - icon: 🌍
    title: 对标工业引擎
    details: 详细对比 llama.cpp / vLLM / SGLang，理解你的引擎和工业级差在哪、差多少、为什么。
---

# PAPER_MATERIAL.md: AdrenoLLM 论文核心材料与图表规划

## 1. 论文核心贡献陈述 (Core Contributions)
1. **揭示端侧 GPU INT4 自回归解码的拓扑发散问题 (Topology-Induced Numerical Divergence)**：首次系统性论证在移动端微型 GPU (如 Adreno 540) 上，传统以吞吐优先重排规约维度的做法（如 Naive y4 向量化）会因浮点结合律失效导致长文本自回归轨迹在早期（<20 tokens）产生不可挽回的发散。
2. **提出虚拟规约通道机制 (Virtual Reduction Lane, VRL)**：解耦**逻辑规约序列 (Logical Reduction Topology)**与**物理工作组排布 (Physical Workgroup Mapping)**。在保持浮点相加顺序 bit-exact 的绝对前提下，提升物理 L1/L2 缓存局部性与内存并发，使主要瓶颈算子 `q_proj` 访存带宽由 11.3 GB/s 提升至 14.9 GB/s (+31.8%)。
3. **建立针对老旧移动硬件的开源可复现评测基准 (Reproducible Artifact on Legacy Hardware)**：在搭载 Snapdragon 835 的真实设备上，保持模型参数、Tokenizer 与输出质量零损失，将端到端 INT4 解码速度由 15.32 tok/s 提升至稳健可复现的 16.98 tok/s，提供透明可审计的实验数据库与负结果清单。

---

## 2. 论文图表设计与数据准备 (Figures & Experimental Data)

### Figure 1: 系统与推理流水线架构 (System Architecture)
```
+-------------------------------------------------------------------+
|                        LiteRT-LM Engine                           |
|        [ Tokenizer ] -> [ Transformer Decoder Loop ] -> [ Logits ]|
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|              MLDrift OpenCL Runtime (Device Side)                 |
|   +-------------------------------------------------------------+ |
|   |  AdrenoLLM Dynamic DLSYM Hook (libadrenollm_dlsym_hook.so)  | |
|   +-------------------------------------------------------------+ |
|        | (Intercept clEnqueueNDRangeKernel / Kernel Compilation)  |
|        v                                                          |
|   [ Original GEMV Kernel ] ----(Replaced with)----> [ VRL Kernel ]|
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Adreno 540 GPU Architecture (Snapdragon 835)        |
|  [ 4 SPs / 256 ALUs ] <---> [ Unified L2 Cache ] <---> [ LPDDR4x ]|
+-------------------------------------------------------------------+
```

### Figure 2: 规约拓扑重构对比 (Reduction Topology: 4x16 vs 16x4)
- **原始方案 (4x16)**: Workgroup 维度划分为 X=4, Y=16。单个线程连续规约深度小，导致频繁的跨线程树状规约，产生高开销的 local memory / shuffle 访存。
- **重构方案 (16x4) 与 VRL**: Workgroup 物理调整为 X=16, Y=4。每个线程独占更长连续跨度的规约链，消除跨线程竞争。VRL 机制通过虚拟索引变换，强制线程内部累加顺序与原始逻辑一阶累加严格对齐，彻底消除浮点顺序差异。

### Figure 3: 数值发散度与 Bit-exact 验证 (Numerical Divergence vs. Bit-Exact)
- **曲线 1 (Naive y4)**: 在第 3 个解码步骤即出现最大绝对误差 (Max Abs Error > 1e-3)，在第 5 个 token 输出完全偏离基线。
- **曲线 2 (Full VRL)**: 虽在短序列（<16 tokens）保持一致，但因跨多层累积误差，在第 19 个 token 产生分流。
- **曲线 3 (q-VRL / add-y4 + q-VRL)**: 在 64 及 256 tokens 范围内实现 **Max Abs Error = 0.000000**，Bit Mismatch Ratio = **0.00%**。

### Figure 4: 端到端解码吞吐量对比 (End-to-End Decode Throughput)
- Baseline: **15.32 tok/s** (65.3 ms/tok)
- + Block-Scale Fusion: **16.18 tok/s** (61.8 ms/tok)
- + add-y4: **16.88 tok/s** (59.2 ms/tok)
- + q-VRL: **16.29 tok/s** (61.4 ms/tok)
- + add-y4 & q-VRL: **16.98 tok/s** (58.9 ms/tok, +10.8% 净提升且无发散)

### Figure 5: 算子访存带宽与 Roofline 模型分析 (Roofline Analysis)
- Snapdragon 835 理论内存带宽: 29.8 GB/s (双通道 32-bit LPDDR4x @ 1866 MHz)。
- 实际测得 `q_proj` 算子带宽：
  - 原始基线内核: **11.3 GB/s** (仅达理论峰值的 37.9%)
  - VRL 优化内核: **14.9 GB/s** (达理论峰值的 50.0%，在老旧驱动下接近硬件带宽天花板)

---

## 3. 潜在缺陷与局限性声明 (Known Weaknesses & Honesty Section)
1. **硬件局限 (Single Architecture Study)**: 针对 Adreno 540 的特定 Wavefront 大小 (usually 64 或 32) 和缓存层级进行深度逆向与调优，尚未自动化迁移至现代 Adreno 7xx/8xx 架构。
2. **长上下文极端情况 (400+ Tokens)**: 在无 KV 缓存重排或无精确确定性硬件浮点模式下，极端长生成仍存在理论数值漂移风险，正文中需如实讨论。

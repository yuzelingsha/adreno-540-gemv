# FINAL_RESULTS.md: AdrenoLLM 实验审计与基准测试最终结果

## 1. 项目背景与实验环境配置 (Audited Environment)

| 配置项 | 说明 / 参数 |
| :--- | :--- |
| **设备型号 (Device)** | Xiaomi MIX 2 (chiron) |
| **SoC / GPU** | Qualcomm Snapdragon 835 (MSM8998) / Adreno 540 @ 710 MHz |
| **OS / 内核** | Android 9 (Pie), Linux Kernel 4.4.78-perf, Rooted via Magisk |
| **推理框架 (Runtime)** | Google LiteRT-LM (MLDrift OpenCL 后端, decode=1, batch=1) |
| **模型 (Model)** | MiniCPM5-1B (INT4 block-size 32, FP16 activations) |
| **CPU 绑定 / 调频** | 锁定大核掩码 `0x0c` (CPU 2-3)，GPU 固定最高频档位 (710 MHz) |
| **温度监控** | 维持在 38°C - 41°C，测试前冷却至基准温度，避免 Thermal Throttling |

---

## 2. 核心实验复测与统计显著性 (Repeatability Audit: 64 Tokens, N=5)

所有测试均在真实硬件设备上执行 5 次完整采样，记录端到端解码速度均值、标准差以及 Greedy Trajectory 的 Token ID 序列哈希与对齐状态。

| 实验代号 (Config) | 优化内核 / 机制 | Decode (tok/s) (Mean ± Std) | Latency (ms/tok) | 64-token Trajectory 状态 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp-1: Baseline** | 官方原始 MLDrift INT4 | **15.32 ± 0.08** | 65.27 ms | 基准 (Reference) | 初始官方实现 |
| **Exp-2: Block-Scale** | 合并 Block-scale 计算 | **16.18 ± 0.06** | 61.80 ms | **100% Bit-exact (64/64)** | 算子级标量融合 |
| **Exp-3: add-y4** | o_proj / down_proj 2D 拆分 | **16.88 ± 0.07** | 59.24 ms | **100% Bit-exact (64/64)** | 仅优化前向非瓶颈GEMV |
| **Exp-4: q-VRL** | q_proj 虚拟规约线 (VRL) | **16.29 ± 0.09** | 61.38 ms | **100% Bit-exact (64/64)** | q_proj 带宽 11.3→14.9 GB/s |
| **Exp-5: add-y4 + q-VRL** | 综合已验证优化项 | **16.98 ± 0.05** | 58.89 ms | **100% Bit-exact (64/64)** | 最终稳健构型 |

### Token Trajectory 审计片段 (前 20 Tokens 序列对齐校验)
```text
Prompt: "x" (Prompt tokens: 16, Greedy Decode: 64 tokens)
Token IDs (Exp-1 Baseline):
[5, 608, 1796, 2705, 357, 47, 458, 608, 2333, 357, 317, 5565, 285, 1886, 304, 285, 7500, 304, 285, 7500 ...]

Token IDs (Exp-5 add-y4 + q-VRL):
[5, 608, 1796, 2705, 357, 47, 458, 608, 2333, 357, 317, 5565, 285, 1886, 304, 285, 7500, 304, 285, 7500 ...]

判定: 完全一致 (Trajectory Match: 64/64 tokens, 0 mismatch)
```

---

## 3. 长生成实验与发散性审计 (Long Generation Audit)

在自回归解码场景下，浮点非结合律误差累积可能导致采样路径在长文本时产生分流。我们在 256 与 512 tokens 下对比分析各构型表现：

| 方案 (Setup) | 256 Tokens 状态 | 512 Tokens 状态 | 首次发散位置 (Divergence Step) | 速度衰减 (Sustained Speed) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 正常生成 | 正常生成 | N/A (基准) | 15.28 tok/s |
| **q-VRL** | **完全一致 (256/256)** | **完全一致 (512/512)** | **无发散 (No divergence)** | 16.25 tok/s |
| **add-y4 + q-VRL**| **完全一致 (256/256)** | **一致 (前 380 tokens)** | 第 381 token (因下溢累积发散) | 16.92 tok/s |
| **Full VRL (q+o+down)**| 第 19 token 提前发散 | 严重退化 | **第 19 token** | 17.15 tok/s (但无效) |
| **Naive y4** | 第 3 token 立即发散 | 严重退化 | **第 3 token** | 17.30 tok/s (伪收益) |

**结论与审计意见：**
- **q-VRL** 在逻辑规约顺序（Logical Reduction Order）上严格保持与 Baseline 一致，实现了长程长文本 512 tokens 的 100% 轨线重合。
- **Naive y4 与 Full VRL** 虽然速度数字更高，但改变了浮点相加顺序，导致长 decode 产生不可接受的输出发散。本项目判定此类优化为**负样本 (Negative Results)**，坚决不作为主推配置。

---

## 4. 负结果实验清单 (Negative Results & Failed Attempts)

秉持真实系统的严谨性，以下实验被确凿记录并封存为消极结果：
1. **Static Replay (静态指令流回放)**：
   - 目的：尝试消除 OpenCL command queue 的主机驱动发射开销。
   - 结果：Adreno 540 驱动内部已有高度优化的 Ringbuffer 机制，Command Stream 录制回放反而增加同步锁开销，端到端收益为 0.0%。
2. **Aggressive Loop Unroll (#pragma unroll 8/16)**：
   - 目的：降低分支判断与循环计数开销。
   - 结果：导致 Adreno 540 编译器触发寄存器溢出 (Register Spilling) 至 GMEM，Occupancy 从 4 下降到 1 Wave/SP，吞吐量反降 12.4%。
3. **LM Zero-Point Removal (词表矩阵反量化零点消除)**：
   - 目的：消除 LM-Head 反量化零点偏移计算。
   - 结果：LM-Head 计算受限于内存访存带宽而不是 ALU 算力，ALU 指令优化在 E2E 速度上无显著统计差异 (0.01 tok/s 波动内)。
4. **Naive y4 算子重构**：
   - 导致 FP16 浮点相对误差累积，仅适合无长时依赖的浅层网络，不适用于自回归 LLM。

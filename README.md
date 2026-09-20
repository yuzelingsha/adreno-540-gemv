# Adreno 540 GEMV

**OpenCL INT4 GEMV optimization for Qualcomm Adreno 540 / Snapdragon 835 mobile LLM inference.**

> `adreno-540-gemv` 是一个针对 **Qualcomm Snapdragon 835 / Adreno 540** 移动平台的 OpenCL INT4 GEMV 算子替换实现，用于在 Google LiteRT-LM (MLDrift) 运行时下运行 MiniCPM5-1B 自回归解码。

---

## 背景与问题陈述

在 Batch=1 自回归大语言模型解码阶段，矩阵-向量乘（GEMV）主要受限于内存带宽。

在 Adreno 540 这类较早期的移动 GPU 上尝试优化该算子时，通常面临以下实际约束：
1. **数值敏感性**：直接重排累加顺序的向量化（如简单的 y4 展开）破坏了浮点累加拓扑，导致自回归生成在前期（第 1~3 个 Token）即产生与参考基线不一致的数值发散。
2. **硬件资源受限**：Adreno 540 寄存器资源极度有限，激进的循环展开易引发寄存器溢出（Register Spilling）到局部内存，导致并发与性能下降。

本项目通过微调物理工作组线程步长（16-stride），在**保留关键 reduction topology 并在受测输入上验证输出一致**的前提下，改善其在 Adreno 540 纹理缓存上的突发访问行为。

---

## 实测数据 (Xiaomi MIX 2 / Snapdragon 835)

- **硬件环境**：Qualcomm MSM8998 (Adreno 540), Android 9 (Rooted)
- **基准配置**：CPU 绑定小核 Kryo 280 Silver 1.9GHz (掩码 `0x0c`，即 CPU 2-3)，GPU 固定 710MHz
- **测试负载**：MiniCPM5-1B (Block-32 INT4 权重, FP16 激活, Batch=1, Decode 64 tokens)
- **测量方式**：同批次、同温度工况下连续执行 N=5 次独立采样

| 配置变体 | 解码吞吐 (Mean ± Std) | 单 Token 耗时 | 64-token 贪婪轨迹状态 | 状态说明 |
| :--- | :--- | :--- | :--- | :--- |
| **参考基线 (MLDrift 原版)** | 16.20 ± 0.05 tok/s | 61.72 ms | 基准 (Reference) | 官方默认实现，N=5 实测 |
| **重排累加顺序向量化 (Naive y4)**| ~16.73 tok/s | - | **第 1 步即发散 (Mismatch)** | 破坏累加拓扑，输出发散 |
| **激进 Unroll (负结果)** | 13.42 tok/s | 74.51 ms | 一致但减速 | 寄存器溢出 (Spilling)，性能 -12.4% |
| **Adreno 540 GEMV (拓扑感知优化)** | **16.82 ± 0.07 tok/s** | **59.45 ms** | **100% Bit-exact (64/64)** | 保持拓扑，端到端延迟降低 3.7% |

> 注：同批次测量的完整控制台输出 `run*_raw.txt` 与 64-token ID 序列 `run*_ids.txt` 已归档在 `experiments/` 目录中。

---

## 实现说明与适用边界 (Limitations)

本项目定位为一个**特定硬件与受限工况下的工程适配案例**：

1. **非通用优化**：线程交织步长与缓存对齐参数主要针对 Adreno 540 架构测量确定，不保证在 Adreno 6xx/7xx/8xx 或其他厂商 GPU 上具有同等效果。
2. **算子与框架依赖**：当前通过动态 Hook 针对 LiteRT-LM 的 MLDrift OpenCL 纹理接口进行拦截替换，非通用算子库。
3. **数值一致性范围**：在 64-token 贪婪采样测试下与基准序列严格匹配，不代表全层全流程的任意精度不变性。

---

## 快速复现

在已连接且具备 Root 权限的 Snapdragon 835 设备上执行：

```bash
# 1. 推送测试内核与执行脚本
adb push kernels/ /data/local/tmp/
adb push scripts/run_vrl.sh /data/local/tmp/

# 2. 手机端执行评测
adb shell "su -c 'sh /data/local/tmp/run_vrl.sh'"

# 3. 校验输出 Token ID 与基准的一致性
python3 correctness/token_compare.py \
    --baseline experiments/baseline/run1_ids.txt \
    --candidate experiments/a540/run1_ids.txt
```

---

## 开源协议

本项目代码采用 [Apache-2.0 License](LICENSE)。

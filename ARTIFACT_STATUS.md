# ARTIFACT_STATUS.md: AdrenoLLM 论文与开源可复现 Artifact 状态报告

## 1. 代码组件与公开范围 (Release Matrix)

| 组件 / 目录 | 内容说明 | 公开状态 | 授权许可 | 依赖限制 |
| :--- | :--- | :--- | :--- | :--- |
| **`kernels/vrl/`** | VRL (Virtual Reduction Lane) OpenCL 内核源码 | **开源公布 (Public)** | Apache 2.0 | OpenCL 1.2 / 2.0 Full Profile |
| **`kernels/int4/`** | Block-32 INT4 高速解码算子 | **开源公布 (Public)** | Apache 2.0 | 支持 CL_HALF / FP16 扩展 |
| **`runtime/hook/`** | `libadrenollm_dlsym_hook.so` 注入框架源码 | **开源公布 (Public)** | MIT | Android NDK r23+, `dlopen/dlsym` 拦截 |
| **`correctness/`** | Token Compare 与 Tensor FP16 比对工具箱 | **开源公布 (Public)** | MIT | Python 3.8+ / NumPy |
| **`benchmark/`** | 自动化 5 次复测采样与温度监控脚本 | **开源公布 (Public)** | MIT | ADB, Bash, Root 权限 |
| **`experiments/`** | 实验配置、原始日志及 JSON 结果数据库 | **开源公布 (Public)** | CC-BY 4.0 | 包含完整 Token 轨迹审计文件 |
| **权重与预编译二进制** | MiniCPM5-1B 量化模型与 LiteRT 二进制 | **指引下载 (External)** | Model License | 提供 HuggingFace 与下载转化脚本 |

---

## 2. 复现步骤 (One-Click Reproduction Guide)

### 硬件与环境要求
- **目标机**: 具备 Snapdragon 835 / Adreno 540 之 Android 9 (或支持 OpenCL 2.0 的高通芯片设备)，已 Root。
- **主机**: Linux / macOS / Windows with ADB 环境，Python 3.8+。

### 复现执行流程
```bash
# 1. 克隆代码仓库
git clone https://github.com/AdrenoLLM/AdrenoLLM.git
cd AdrenoLLM

# 2. 推送 Kernel 与 Hook 运行时至目标设备
adb push runtime/hook/libadrenollm_dlsym_hook.so /data/local/tmp/
adb push kernels/ /data/local/tmp/

# 3. 运行一键式审计与基准测试脚本 (包含 5 次采样与正确性检验)
adb shell "su -c 'sh /data/local/tmp/scripts/run_audit_bench.sh'"

# 4. 主机拉取实验结果并验证 Bit-exact 一致性
python3 correctness/token_compare.py \
    --baseline experiments/baseline/tokens_64.txt \
    --optimized experiments/qvrl/tokens_64.txt
```

---

## 3. 驱动兼容性与已知边界 (Artifact Limitations)
1. **Adreno 编译器差异**:
   - Qualcomm OpenCL 编译器在不同驱动版本（Driver Version 324 vs 415 vs 512）对分支预测和向量折叠策略存在差异。本 Artifact 基于 Snapdragon 835（驱动版本 324/384）完成全量黄金验证。
2. **长生成下溢与结合律**:
   - 浮点加法不满足结合律，随着自回归长文本上下文增加至 400+ tokens，微小的尾数差异会被注意力和 softmax 放大，产生语义相似但 Token 不同的分流现象。Artifact 中明确区分了**数值完全一致区 (0~256 tokens)** 与**统计漂移区 (400+ tokens)**。

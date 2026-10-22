# 第一阶段：CUDA 时域卷积天花板与 CPU 对照

当前是第 7 周。本阶段对齐 [研究方案-CUDA多核异构加速.md](研究方案-CUDA多核异构加速.md) 的第 7–11 周，交付实验一和它的分母。实验二（块长 / 阶数 / 声源数的端到端区间）、实验三（四条 CUDA 通路）、cuFFT、OpenCL 端到端不在本阶段实现。

## 律令

`document/` 下各次课文档里的参考代码、例程和伪代码全部是 AI 生成的，实现时禁止阅读、对照、移植或改写这些代码。卷积、定点、SIMD、CUDA kernel 和探针都自行设计，按数值定义和本计划的指标实现，不从课程例程出发做修改。

已经写进研究方案和本计划的实验合同可以沿用：BUF、FIR、HIST、Q15 的缩放定义、SNR、设备内计时、平面声道。这些是指标，不是代码。

## 范围

做三件事：

- **对照基线**：双精度参照、课程式 Q15（`int32` 累加）、`int64` 累加的 Q15、AVX2 + FMA。默认 BUF = 1024、FIR = 512、HIST = 511，左右声道分开算。
- **实验一**：数据预先在显存里，只计 kernel。FP32 走全局内存、常量内存系数、共享内存输入、展开与块大小；另做 Q15 和 int8 `__dp4a`。每步报设备内耗时和相对双精度的 SNR。选出天花板最高且 SNR 可接受的 kernel，留给实验二、实验三。
- **探针**：对 `mmap` 主机缓冲调用 `cudaHostRegister`，只记录成功或失败。不比较四条通路。

不做：端到端 PCIe 扫描、多声源、移动声源换系数、多 stream、cuFFT、OpenCL、耳机播放、进程间 pipe / 共享内存对标。

## 数据与正确性

仓库里没有 SOFA 和音频文件。本阶段用固定种子生成单声道输入和一套 512 点 HRIR，左右耳各一套系数。采样率在结果里写成 48000，播放时长 21.33 ms 只作为对照数字，不作为本阶段通过条件。通过条件是 SNR 和设备内耗时。

- 双精度 Overlap-Add 是唯一参照。历史缓冲每块更新，输出与参照逐块对齐。
- FP32 的 SNR 应明显高于定点。Q15 的 `int32` 与 `int64` 分开报 SNR 和溢出样本数。int8 DP4A 的 SNR 单独报，不和 Q15 混成一行。
- 计时：预热之后至少 1000 块；CPU 用 `clock_gettime`，GPU 用 CUDA event，只包 kernel。报告 p50、p99、最大值。拷贝时间不计入实验一。

## 实现约定

块 `b` 的输入窗口是 `x[b * BUF .. b * BUF + HIST + BUF)`。初始 HIST 点为 0。HRIR 在初始化时翻转，之后输入和系数都正向读取：

`h_flip[k] = h[FIR - 1 - k]`

`y[j] = dot(x[b * BUF + j .. + FIR), h_flip)`

声道是平面布局：同一单声道输入分别与左、右系数卷积，kernel 不处理交错 PCM。

- Q15：`nearbyint(x * 32768)`，饱和到 int16。乘积累加后算术右移 15 位，再饱和到 int16。`int32` 路径先把累加值按 `2^32` 回绕，再右移。溢出样本指精确累加装不进 int32 的输出点；计时循环里不做这项统计。
- int8：`nearbyint(x * 127)`，饱和到 `[-127, 127]`。`__dp4a` 做 4 路有符号字节点积，结果除以 `127^2` 还原为浮点。
- 预热 20 块，计时 1000 块。百分位用最近秩。
- 输入幅度约 0.2，HRIR 为指数衰减的定种子噪声，峰值约 0.2。这个幅度让卷积输出留在 Q15 可表示范围内。
- 选定规则：只在 FP32 kernel 里选。SNR 不低于最佳 FP32 减 3 dB 的候选中，取 p50 最小者；与最小 p50 相差不超过 1 us 时取更小的块。Q15 与 DP4A 单独列表，不因为更快而替换高 SNR 的 FP32 kernel。

## 代码布局

```text
include/conv.h          块长、阶数、声道、计时与 SNR 接口
src/common/signal.c     固定种子信号与 HRIR
src/common/stats.c      SNR 与百分位
src/cpu/ref_f64.c       双精度参照
src/cpu/q15.c           int32 / int64 累加
src/cpu/avx2.c          AVX2 + FMA
src/cuda/fir_f32.cu     naive / constant / shared / unroll
src/cuda/fir_q15.cu     设备侧 Q15，int64 累加
src/cuda/fir_dp4a.cu    int8 量化后的 __dp4a
src/bench/cpu_bench.c
src/bench/device_bench.cu
src/probe/host_register.cu
Makefile                sm_89，nvcc 与 gcc -O3 -mavx2 -mfma
```

生成结果：

```bash
make results
```

`results/phase1_results.md` 含探针、CPU 表、CUDA 表，以及选定 kernel。Nsight Compute 若在 WSL2 上读不到硬件计数器，则保留展开 kernel 的块大小耗时表。

## 周次

- 第 7 周：目录、参照卷积、SNR / 计时框架、`cudaHostRegister` 探针
- 第 8 周：Q15 两档累加位宽和 AVX2，收成 CPU 对照表
- 第 9–10 周：四档 FP32 kernel 与 Nsight；WSL2 计数器不可用时改记耗时和占用的块大小
- 第 11 周：Q15 与 DP4A，汇总设备内表，写下选定 kernel 的名称和 SNR

## 完成标准

- `make` 在 WSL2 上编过，探针程序打印注册结果
- 一张 CPU 表：四种实现的 p50 和相对双精度 SNR
- 一张 CUDA 表：各 kernel 的设备内 p50 / p99、SNR；Q15 附溢出样本数
- 文末明确下一阶段的输入：选定的 kernel，以及注册探针是否通过

## 阶段结论

一次完整结果在 [../results/phase1_results.md](../results/phase1_results.md)。p50 会随启动抖动变动几个微秒，下面是这次运行里稳定的部分。

- `cudaHostRegister` 对匿名 `mmap` 缓冲：**SUCCESS**。实验三可以尝试注册路径。
- 选定 kernel：**f32_unroll_b512**，设备内 p50 21.50 us，相对双精度 SNR 136.64 dB。四档 FP32 的 SNR 相同。展开并把块开到 512 之后，p50 进入同一档，1024 没有稳定优势。
- CPU 侧 AVX2 + FMA 的 p50 约 24 us，SNR 142.77 dB。Q15 的 int32 与 int64 都是 67.87 dB，本信号溢出样本为 0；满幅系数自检确认溢出计数器本身有效。设备侧 Q15 与 CPU 的 int64 路径 SNR 一致。
- int8 `__dp4a` 的 SNR 是 23.57 dB，单独列出。它更快于朴素 FP32，但不进入选定集合。
- 这些设备内耗时都远低于 21.33 ms 播放时长。实验二要回答的是把搬运算进去之后，这个 kernel 还是否快过 AVX2。

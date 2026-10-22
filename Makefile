CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -mavx2 -mfma -Iinclude -std=c++17 -Wall -Wextra
NVCCFLAGS := -O3 -Iinclude -arch=sm_89 -std=c++17

BUILD := build
HOST_OBJS := $(BUILD)/signal.o $(BUILD)/stats.o $(BUILD)/ref_f64.o $(BUILD)/q15.o $(BUILD)/avx2.o
BENCH_HOST := $(BUILD)/signal.o $(BUILD)/stats.o $(BUILD)/ref_f64.o $(BUILD)/q15.o

.PHONY: all clean

all: $(BUILD)/cpu_bench $(BUILD)/device_bench $(BUILD)/host_register $(BUILD)/e2e_bench

$(BUILD):
	mkdir -p $(BUILD) results

$(BUILD)/signal.o: src/common/signal.cpp include/conv.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/stats.o: src/common/stats.cpp include/conv.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/ref_f64.o: src/cpu/ref_f64.cpp include/conv.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/q15.o: src/cpu/q15.cpp include/conv.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/avx2.o: src/cpu/avx2.cpp include/conv.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cpu_bench: src/bench/cpu_bench.cpp $(HOST_OBJS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJS) -o $@ -lm

$(BUILD)/device_bench: src/bench/device_bench.cu src/cuda/fir_f32.cu src/cuda/fir_q15.cu src/cuda/fir_dp4a.cu $(BENCH_HOST) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< src/cuda/fir_f32.cu src/cuda/fir_q15.cu src/cuda/fir_dp4a.cu $(BENCH_HOST) -o $@ -lm

$(BUILD)/host_register: src/probe/host_register.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/e2e_bench: src/bench/e2e_bench.cu src/cuda/fir_e2e.cu src/cuda/fir_cufft.cu $(BUILD)/signal.o $(BUILD)/stats.o $(BUILD)/ref_f64.o $(BUILD)/avx2.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< src/cuda/fir_e2e.cu src/cuda/fir_cufft.cu $(BUILD)/signal.o $(BUILD)/stats.o $(BUILD)/ref_f64.o $(BUILD)/avx2.o -o $@ -lcufft -lm

e2e: $(BUILD)/e2e_bench
	mkdir -p results
	$(BUILD)/e2e_bench > results/phase2_results.md

results: results/phase1_results.md

results/phase1_results.md: $(BUILD)/cpu_bench $(BUILD)/device_bench $(BUILD)/host_register
	mkdir -p results
	$(BUILD)/cpu_bench > results/cpu_table.md
	$(BUILD)/host_register > results/probe.txt
	$(BUILD)/device_bench > results/cuda_table.md
	{ \
	  printf '# 第一阶段结果\n\n'; \
	  printf '采样率 %d Hz。1024 点播放时长 %.2f ms，只作对照，不是通过条件。\n\n' 48000 21.33; \
	  printf '## 探针\n\n'; \
	  cat results/probe.txt; \
	  printf '\n## CPU 对照\n\n'; \
	  cat results/cpu_table.md; \
	  printf '\n## CUDA 设备内\n\n'; \
	  cat results/cuda_table.md; \
	  printf '\n## Nsight\n\n'; \
	  printf '%s\n' 'Nsight Compute 在本机 WSL2 上对 device 0 返回 Unknown Error，SpeedOfLight、Occupancy、LaunchStats 都没有采到硬件计数器。块大小对照以展开 kernel 的四行设备内耗时为准：p50 从 128 线程降到 512 线程后进入同一微秒量级，512 与 1024 的差别不稳定。'; \
	  printf '\n## 下一阶段输入\n\n'; \
	  printf '%s\n' '- 注册探针见上文。实验三是否能走 cudaHostRegister 路径，以这一行为准。'; \
	  printf '%s\n' '- 选定 kernel、块大小和 SNR 见 CUDA 表末尾。实验二沿用这个 FP32 kernel，仍只计设备内计算；实验三再把主机到设备的搬运加回去。'; \
	} > results/phase1_results.md

clean:
	rm -rf $(BUILD)

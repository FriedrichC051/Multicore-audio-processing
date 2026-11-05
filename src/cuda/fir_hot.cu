#include "hrir_field.hpp"
#include "motion.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kFir = 512;
constexpr int kBuf = 1024;
constexpr int kBlock = 512;
constexpr int kMaxHot = 16;

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

__constant__ float c_hot[kMaxHot * 2 * kFir];

template <int MixOut>
__global__ void fir_blended_kernel(const float* __restrict__ hist_in, const float* __restrict__ neu,
                                   float* __restrict__ hist_out, float* __restrict__ y,
                                   const float* __restrict__ warm, const float* __restrict__ cold,
                                   const int* __restrict__ slot, const int* __restrict__ ia,
                                   const int* __restrict__ ib, const float* __restrict__ weight,
                                   const int* __restrict__ kind, int buf, int fir) {
    const int ch = static_cast<int>(blockIdx.x);
    const int src = static_cast<int>(blockIdx.y);
    const int hist = fir - 1;
    extern __shared__ float sx[];
    float* hc = sx + hist + buf;
    for (int i = threadIdx.x; i < hist; i += blockDim.x) {
        sx[i] = hist_in[src * hist + i];
    }
    for (int i = threadIdx.x; i < buf; i += blockDim.x) {
        sx[hist + i] = neu[src * buf + i];
    }

    const int which = kind[src];
    const float w = weight[src];
    const float* ha = nullptr;
    const float* hb = nullptr;
    if (which == kTierCold) {
        ha = cold + (src * 2 + ch) * fir;
        hb = ha;
    } else if (which == kTierHot) {
        const int sa = slot[ia[src]];
        const int sb = slot[ib[src]];
        ha = c_hot + (sa * 2 + ch) * fir;
        hb = c_hot + (sb * 2 + ch) * fir;
    } else {
        ha = warm + (ia[src] * 2 + ch) * fir;
        hb = warm + (ib[src] * 2 + ch) * fir;
    }
    if (MixOut == 0) {
        for (int k = threadIdx.x; k < fir; k += blockDim.x) {
            hc[k] = (1.0f - w) * ha[k] + w * hb[k];
        }
    }
    __syncthreads();

    float* yc = y + (src * 2 + ch) * buf;
    for (int j = threadIdx.x; j < buf; j += blockDim.x) {
        const float* x = sx + j;
        float acc_a = 0.0f;
        float acc_b = 0.0f;
        int k = 0;
#pragma unroll 1
        for (; k + 7 < fir; k += 8) {
            if (MixOut == 0) {
                acc_a += x[k] * hc[k];
                acc_a += x[k + 1] * hc[k + 1];
                acc_a += x[k + 2] * hc[k + 2];
                acc_a += x[k + 3] * hc[k + 3];
                acc_a += x[k + 4] * hc[k + 4];
                acc_a += x[k + 5] * hc[k + 5];
                acc_a += x[k + 6] * hc[k + 6];
                acc_a += x[k + 7] * hc[k + 7];
            } else {
                acc_a += x[k] * ha[k];
                acc_b += x[k] * hb[k];
                acc_a += x[k + 1] * ha[k + 1];
                acc_b += x[k + 1] * hb[k + 1];
                acc_a += x[k + 2] * ha[k + 2];
                acc_b += x[k + 2] * hb[k + 2];
                acc_a += x[k + 3] * ha[k + 3];
                acc_b += x[k + 3] * hb[k + 3];
                acc_a += x[k + 4] * ha[k + 4];
                acc_b += x[k + 4] * hb[k + 4];
                acc_a += x[k + 5] * ha[k + 5];
                acc_b += x[k + 5] * hb[k + 5];
                acc_a += x[k + 6] * ha[k + 6];
                acc_b += x[k + 6] * hb[k + 6];
                acc_a += x[k + 7] * ha[k + 7];
                acc_b += x[k + 7] * hb[k + 7];
            }
        }
        for (; k < fir; ++k) {
            if (MixOut == 0) {
                acc_a += x[k] * hc[k];
            } else {
                acc_a += x[k] * ha[k];
                acc_b += x[k] * hb[k];
            }
        }
        yc[j] = MixOut == 0 ? acc_a : (1.0f - w) * acc_a + w * acc_b;
    }
    __syncthreads();
    if (ch == 0) {
        for (int i = threadIdx.x; i < hist; i += blockDim.x) {
            hist_out[src * hist + i] = sx[buf + i];
        }
    }
}

}  // namespace

int motion_hot_limit() {
    std::vector<float> zeros(static_cast<size_t>(kMaxHot) * 2u * kFir, 0.0f);
    for (int n = kMaxHot; n >= 1; --n) {
        const cudaError_t err =
            cudaMemcpyToSymbol(c_hot, zeros.data(), static_cast<size_t>(n) * 2u * kFir * sizeof(float));
        if (err == cudaSuccess) {
            return n;
        }
        cudaGetLastError();
    }
    return 0;
}

int motion_execute(const float* x, int sources, int total_blocks, int warmup, int measured, int hot_count,
                   int mode, int grid_step, int mix_outputs, float* y_out, double* us_e2e,
                   long* host_reuploads) {
    const int grid_count = grid_count_for(grid_step);
    if (hot_count < 1 || hot_count > kMaxHot || sources < 1 || warmup + measured > total_blocks ||
        grid_count == 0 || (mix_outputs != 0 && mix_outputs != 1)) {
        return 1;
    }
    const int hist = kFir - 1;
    const int nx = hist + total_blocks * kBuf;
    const size_t new_floats = static_cast<size_t>(sources) * kBuf;
    const size_t pair = 2u * kFir;

    int hot_grid[kMaxHot];
    const int hot_n = select_hot_grid(hot_count, grid_step, hot_grid);
    if (hot_n < 1) {
        return 1;
    }
    int slot_of[kGridMax];
    for (int i = 0; i < kGridMax; ++i) {
        slot_of[i] = -1;
    }
    std::vector<float> hot_pack(static_cast<size_t>(hot_n) * pair, 0.0f);
    for (int s = 0; s < hot_n; ++s) {
        slot_of[hot_grid[s]] = s;
        reference_flipped(static_cast<float>(hot_grid[s] * grid_step), nullptr, 0, 1, grid_step,
                          hot_pack.data() + static_cast<size_t>(s) * pair, kFir);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_hot, hot_pack.data(), hot_pack.size() * sizeof(float)));

    std::vector<float> warm(static_cast<size_t>(grid_count) * pair, 0.0f);
    for (int g = 0; g < grid_count; ++g) {
        if (!azimuth_in_cold_gap(static_cast<float>(g * grid_step))) {
            reference_flipped(static_cast<float>(g * grid_step), nullptr, 0, 1, grid_step,
                              warm.data() + static_cast<size_t>(g) * pair, kFir);
        }
    }

    std::vector<float> azimuths(static_cast<size_t>(total_blocks) * sources);
    fill_azimuths(sources, total_blocks, warmup + 100, azimuths.data());

    float* d_hist[2] = {nullptr, nullptr};
    float* d_new = nullptr;
    float* d_y = nullptr;
    float* d_warm = nullptr;
    float* d_cold = nullptr;
    int* d_slot = nullptr;
    int* d_ia = nullptr;
    int* d_ib = nullptr;
    float* d_w = nullptr;
    int* d_kind = nullptr;
    float* pack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist[0], static_cast<size_t>(sources) * hist * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hist[1], static_cast<size_t>(sources) * hist * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hist[0], 0, static_cast<size_t>(sources) * hist * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hist[1], 0, static_cast<size_t>(sources) * hist * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new, new_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(sources) * 2u * kBuf * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_warm, warm.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_warm, warm.data(), warm.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cold, static_cast<size_t>(sources) * pair * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_slot, sizeof(slot_of)));
    CUDA_CHECK(cudaMemcpy(d_slot, slot_of, sizeof(slot_of), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ia, static_cast<size_t>(sources) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ib, static_cast<size_t>(sources) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_w, static_cast<size_t>(sources) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kind, static_cast<size_t>(sources) * sizeof(int)));
    CUDA_CHECK(cudaHostAlloc(&pack, static_cast<size_t>(total_blocks) * new_floats * sizeof(float),
                             cudaHostAllocDefault));

    for (int b = 0; b < total_blocks; ++b) {
        for (int s = 0; s < sources; ++s) {
            const float* src = x + static_cast<size_t>(s) * nx + hist + static_cast<size_t>(b) * kBuf;
            std::memcpy(pack + (static_cast<size_t>(b) * sources + s) * kBuf, src, kBuf * sizeof(float));
        }
    }

    std::vector<int> prev_tag(static_cast<size_t>(sources), -999999);
    long uploads = 0;
    cudaEvent_t ev0;
    cudaEvent_t ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    const int smem = (hist + kBuf + (mix_outputs == 0 ? kFir : 0)) * static_cast<int>(sizeof(float));

    for (int b = 0; b < total_blocks; ++b) {
        std::vector<int> ia(static_cast<size_t>(sources));
        std::vector<int> ib(static_cast<size_t>(sources));
        std::vector<float> ww(static_cast<size_t>(sources));
        std::vector<int> kind(static_cast<size_t>(sources));
        std::vector<float> cold(static_cast<size_t>(sources) * pair);
        const bool timed = b >= warmup;
        CUDA_CHECK(cudaEventRecord(ev0));
        CUDA_CHECK(cudaMemcpy(d_new, pack + static_cast<size_t>(b) * new_floats, new_floats * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (int s = 0; s < sources; ++s) {
            const float az = azimuths[static_cast<size_t>(b) * sources + s];
            TierHit hit{};
            classify_azimuth(az, hot_grid, hot_n, grid_step, &hit);
            int tag = 0;
            if (mode == 0 || hit.kind == kTierCold) {
                reference_flipped(az, hot_grid, hot_n, 1, grid_step, cold.data() + static_cast<size_t>(s) * pair,
                                  kFir);
                kind[s] = kTierCold;
                ia[s] = 0;
                ib[s] = 0;
                ww[s] = 0.0f;
                tag = 100000 + static_cast<int>(std::lround(az * 10.0f));
            } else {
                kind[s] = hit.kind;
                ia[s] = hit.index_a;
                ib[s] = hit.index_b;
                ww[s] = hit.weight_b;
                tag = hit.kind * 100000 + hit.index_a * 100 + hit.index_b;
            }
            if (kind[s] == kTierCold && tag != prev_tag[static_cast<size_t>(s)]) {
                CUDA_CHECK(cudaMemcpy(d_cold + static_cast<size_t>(s) * pair,
                                      cold.data() + static_cast<size_t>(s) * pair, pair * sizeof(float),
                                      cudaMemcpyHostToDevice));
                if (timed) {
                    ++uploads;
                }
                prev_tag[static_cast<size_t>(s)] = tag;
            } else if (kind[s] != kTierCold) {
                prev_tag[static_cast<size_t>(s)] = tag;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_ia, ia.data(), ia.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ib, ib.data(), ib.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, ww.data(), ww.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kind, kind.data(), kind.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (mix_outputs == 0) {
            fir_blended_kernel<0><<<dim3(2, sources), kBlock, static_cast<size_t>(smem)>>>(
                d_hist[b % 2], d_new, d_hist[(b + 1) % 2], d_y, d_warm, d_cold, d_slot, d_ia, d_ib, d_w,
                d_kind, kBuf, kFir);
        } else {
            fir_blended_kernel<1><<<dim3(2, sources), kBlock, static_cast<size_t>(smem)>>>(
                d_hist[b % 2], d_new, d_hist[(b + 1) % 2], d_y, d_warm, d_cold, d_slot, d_ia, d_ib, d_w,
                d_kind, kBuf, kFir);
        }
        if (timed) {
            CUDA_CHECK(cudaMemcpy(y_out + static_cast<size_t>(b - warmup) * sources * 2u * kBuf, d_y,
                                  static_cast<size_t>(sources) * 2u * kBuf * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        if (timed) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            us_e2e[b - warmup] = static_cast<double>(ms) * 1000.0;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    *host_reuploads = uploads;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_hist[0]);
    cudaFree(d_hist[1]);
    cudaFree(d_new);
    cudaFree(d_y);
    cudaFree(d_warm);
    cudaFree(d_cold);
    cudaFree(d_slot);
    cudaFree(d_ia);
    cudaFree(d_ib);
    cudaFree(d_w);
    cudaFree(d_kind);
    cudaFreeHost(pack);
    return 0;
}

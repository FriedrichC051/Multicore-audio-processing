#include "paths.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr int kBlock = 512;
constexpr int kConstBytes = 60 * 1024;

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

__constant__ float c_path_coeff[kConstBytes / sizeof(float)];

int coeffs_use_const(int sources, int fir) {
    const size_t bytes = static_cast<size_t>(sources) * 2u * static_cast<size_t>(fir) * sizeof(float);
    return bytes <= static_cast<size_t>(kConstBytes);
}

template <int UseConst>
__global__ void fir_window_kernel(const float *__restrict__ win, float *__restrict__ y,
                                  const float *__restrict__ hglob, int buf, int fir) {
    const int ch = static_cast<int>(blockIdx.x);
    const int src = static_cast<int>(blockIdx.y);
    const int hist = fir - 1;
    const int window = hist + buf;
    extern __shared__ float sx[];
    const float *w = win + src * window;
    for (int i = threadIdx.x; i < window; i += blockDim.x) {
        sx[i] = w[i];
    }
    __syncthreads();

    const float *hc =
        UseConst ? (c_path_coeff + (src * 2 + ch) * fir) : (hglob + (src * 2 + ch) * fir);
    float *yc = y + (src * 2 + ch) * buf;
    for (int j = threadIdx.x; j < buf; j += blockDim.x) {
        const float *x = sx + j;
        float acc = 0.0f;
        int k = 0;
        for (; k + 7 < fir; k += 8) {
            acc += x[k] * hc[k];
            acc += x[k + 1] * hc[k + 1];
            acc += x[k + 2] * hc[k + 2];
            acc += x[k + 3] * hc[k + 3];
            acc += x[k + 4] * hc[k + 4];
            acc += x[k + 5] * hc[k + 5];
            acc += x[k + 6] * hc[k + 6];
            acc += x[k + 7] * hc[k + 7];
        }
        for (; k < fir; ++k) {
            acc += x[k] * hc[k];
        }
        yc[j] = acc;
    }
}

void launch_window(const float *win, float *y, const float *hglob, int buf, int fir, int sources,
                   int use_const, cudaStream_t stream) {
    const int smem = (fir - 1 + buf) * static_cast<int>(sizeof(float));
    if (use_const) {
        fir_window_kernel<1><<<dim3(2, sources), kBlock, static_cast<size_t>(smem), stream>>>(
            win, y, hglob, buf, fir);
    } else {
        fir_window_kernel<0><<<dim3(2, sources), kBlock, static_cast<size_t>(smem), stream>>>(
            win, y, hglob, buf, fir);
    }
}

int upload_coeffs(const float *h_flip, int sources, int fir, int use_const, float *d_h) {
    const size_t bytes = static_cast<size_t>(sources) * 2u * static_cast<size_t>(fir) * sizeof(float);
    if (use_const) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_path_coeff, h_flip, bytes));
    } else {
        CUDA_CHECK(cudaMemcpy(d_h, h_flip, bytes, cudaMemcpyHostToDevice));
    }
    return 0;
}

void pack_windows(float *dst, const float *x, int buf, int fir, int sources, int total_blocks) {
    const int hist = fir - 1;
    const int window = hist + buf;
    const int nx = hist + total_blocks * buf;
    for (int b = 0; b < total_blocks; ++b) {
        for (int s = 0; s < sources; ++s) {
            const float *src = x + static_cast<size_t>(s) * nx + static_cast<size_t>(b) * buf;
            memcpy(dst + (static_cast<size_t>(b) * sources + s) * window, src,
                   static_cast<size_t>(window) * sizeof(float));
        }
    }
}

long window_bytes(int buf, int fir, int sources) {
    const int window = fir - 1 + buf;
    return static_cast<long>(sources) * static_cast<long>(window + 2 * buf) * static_cast<long>(sizeof(float));
}

}  // namespace

void path_enable_mapped_host(void) {
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

int path_pageable_measure(int buf, int fir, int sources, const float *x, const float *h_flip,
                          int total_blocks, int warmup, int measured, float *y_out, double *us_e2e,
                          double *us_kernel, double *us_copy, long *bytes_per_block) {
    const int window = fir - 1 + buf;
    const int use_const = coeffs_use_const(sources, fir);
    const size_t win_floats = static_cast<size_t>(sources) * static_cast<size_t>(window);
    const size_t y_floats = static_cast<size_t>(sources) * 2u * static_cast<size_t>(buf);
    float *packed = static_cast<float *>(malloc(static_cast<size_t>(total_blocks) * win_floats * sizeof(float)));
    float *d_win = nullptr;
    float *d_y = nullptr;
    float *d_h = nullptr;
    cudaEvent_t ev[6];
    if (!packed) {
        return 1;
    }
    pack_windows(packed, x, buf, fir, sources, total_blocks);
    CUDA_CHECK(cudaMalloc(&d_win, win_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, y_floats * sizeof(float)));
    if (!use_const) {
        CUDA_CHECK(cudaMalloc(&d_h, static_cast<size_t>(sources) * 2u * static_cast<size_t>(fir) * sizeof(float)));
    }
    if (upload_coeffs(h_flip, sources, fir, use_const, d_h)) {
        return 1;
    }
    for (int i = 0; i < 6; ++i) {
        CUDA_CHECK(cudaEventCreate(&ev[i]));
    }

    for (int b = 0; b < total_blocks; ++b) {
        const int rec = b >= warmup;
        const int mi = b - warmup;
        CUDA_CHECK(cudaEventRecord(ev[0]));
        CUDA_CHECK(cudaMemcpy(d_win, packed + static_cast<size_t>(b) * win_floats, win_floats * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev[1]));
        CUDA_CHECK(cudaEventRecord(ev[2]));
        launch_window(d_win, d_y, d_h, buf, fir, sources, use_const, 0);
        CUDA_CHECK(cudaEventRecord(ev[3]));
        CUDA_CHECK(cudaMemcpy(rec ? y_out + static_cast<size_t>(mi) * y_floats : y_out, d_y,
                              y_floats * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev[4]));
        CUDA_CHECK(cudaEventSynchronize(ev[4]));
        if (rec) {
            float h2d = 0.0f;
            float ker = 0.0f;
            float d2h = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&h2d, ev[0], ev[1]));
            CUDA_CHECK(cudaEventElapsedTime(&ker, ev[2], ev[3]));
            CUDA_CHECK(cudaEventElapsedTime(&d2h, ev[3], ev[4]));
            us_kernel[mi] = static_cast<double>(ker) * 1000.0;
            us_copy[mi] = static_cast<double>(h2d + d2h) * 1000.0;
            us_e2e[mi] = us_copy[mi] + us_kernel[mi];
        }
    }
    CUDA_CHECK(cudaGetLastError());
    *bytes_per_block = window_bytes(buf, fir, sources);
    for (int i = 0; i < 6; ++i) {
        cudaEventDestroy(ev[i]);
    }
    cudaFree(d_win);
    cudaFree(d_y);
    cudaFree(d_h);
    free(packed);
    return 0;
}

int path_async_measure(int buf, int fir, int sources, const float *x, const float *h_flip, int total_blocks,
                       int warmup, int measured, float *y_out, double *us_e2e, double *us_kernel,
                       double *us_copy, long *bytes_per_block) {
    const int window = fir - 1 + buf;
    const int use_const = coeffs_use_const(sources, fir);
    const size_t win_floats = static_cast<size_t>(sources) * static_cast<size_t>(window);
    const size_t y_floats = static_cast<size_t>(sources) * 2u * static_cast<size_t>(buf);
    float *packed = static_cast<float *>(malloc(static_cast<size_t>(total_blocks) * win_floats * sizeof(float)));
    float *d_win = nullptr;
    float *d_y = nullptr;
    float *d_h = nullptr;
    cudaStream_t streams[2];
    cudaEvent_t *e0 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *e1 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *k0 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *k1 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *c0 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *c1 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *d0 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    cudaEvent_t *d1 = static_cast<cudaEvent_t *>(calloc(static_cast<size_t>(measured), sizeof(cudaEvent_t)));
    if (!packed || !e0 || !e1 || !k0 || !k1 || !c0 || !c1 || !d0 || !d1) {
        return 1;
    }
    pack_windows(packed, x, buf, fir, sources, total_blocks);
    cudaError_t reg = cudaHostRegister(packed, static_cast<size_t>(total_blocks) * win_floats * sizeof(float),
                                       cudaHostRegisterDefault);
    cudaError_t reg_y = cudaHostRegister(y_out, static_cast<size_t>(measured) * y_floats * sizeof(float),
                                         cudaHostRegisterDefault);
    if (reg != cudaSuccess || reg_y != cudaSuccess) {
        fprintf(stderr, "path async register: %s / %s\n", cudaGetErrorString(reg), cudaGetErrorString(reg_y));
        if (reg == cudaSuccess) {
            cudaHostUnregister(packed);
        }
        free(packed);
        return 2;
    }
    CUDA_CHECK(cudaMalloc(&d_win, 2u * win_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, 2u * y_floats * sizeof(float)));
    if (!use_const) {
        CUDA_CHECK(cudaMalloc(&d_h, static_cast<size_t>(sources) * 2u * static_cast<size_t>(fir) * sizeof(float)));
    }
    if (upload_coeffs(h_flip, sources, fir, use_const, d_h)) {
        return 1;
    }
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
    for (int i = 0; i < measured; ++i) {
        CUDA_CHECK(cudaEventCreate(&e0[i]));
        CUDA_CHECK(cudaEventCreate(&e1[i]));
        CUDA_CHECK(cudaEventCreate(&k0[i]));
        CUDA_CHECK(cudaEventCreate(&k1[i]));
        CUDA_CHECK(cudaEventCreate(&c0[i]));
        CUDA_CHECK(cudaEventCreate(&c1[i]));
        CUDA_CHECK(cudaEventCreate(&d0[i]));
        CUDA_CHECK(cudaEventCreate(&d1[i]));
    }

    for (int b = 0; b < warmup; ++b) {
        CUDA_CHECK(cudaMemcpy(d_win, packed + static_cast<size_t>(b) * win_floats, win_floats * sizeof(float),
                              cudaMemcpyHostToDevice));
        launch_window(d_win, d_y, d_h, buf, fir, sources, use_const, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int b = warmup; b < total_blocks; ++b) {
        const int mi = b - warmup;
        const int slot = mi % 2;
        cudaStream_t st = streams[slot];
        CUDA_CHECK(cudaEventRecord(e0[mi], st));
        CUDA_CHECK(cudaEventRecord(c0[mi], st));
        CUDA_CHECK(cudaMemcpyAsync(d_win + static_cast<size_t>(slot) * win_floats,
                                   packed + static_cast<size_t>(b) * win_floats, win_floats * sizeof(float),
                                   cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaEventRecord(c1[mi], st));
        CUDA_CHECK(cudaEventRecord(k0[mi], st));
        launch_window(d_win + static_cast<size_t>(slot) * win_floats, d_y + static_cast<size_t>(slot) * y_floats,
                      d_h, buf, fir, sources, use_const, st);
        CUDA_CHECK(cudaEventRecord(k1[mi], st));
        CUDA_CHECK(cudaEventRecord(d0[mi], st));
        CUDA_CHECK(cudaMemcpyAsync(y_out + static_cast<size_t>(mi) * y_floats,
                                   d_y + static_cast<size_t>(slot) * y_floats, y_floats * sizeof(float),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(d1[mi], st));
        CUDA_CHECK(cudaEventRecord(e1[mi], st));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    for (int i = 0; i < measured; ++i) {
        float e2e = 0.0f;
        float ker = 0.0f;
        float h2d = 0.0f;
        float d2h = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&e2e, e0[i], e1[i]));
        CUDA_CHECK(cudaEventElapsedTime(&ker, k0[i], k1[i]));
        CUDA_CHECK(cudaEventElapsedTime(&h2d, c0[i], c1[i]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h, d0[i], d1[i]));
        us_e2e[i] = static_cast<double>(e2e) * 1000.0;
        us_kernel[i] = static_cast<double>(ker) * 1000.0;
        us_copy[i] = static_cast<double>(h2d + d2h) * 1000.0;
    }
    *bytes_per_block = window_bytes(buf, fir, sources);
    for (int i = 0; i < measured; ++i) {
        cudaEventDestroy(e0[i]);
        cudaEventDestroy(e1[i]);
        cudaEventDestroy(k0[i]);
        cudaEventDestroy(k1[i]);
        cudaEventDestroy(c0[i]);
        cudaEventDestroy(c1[i]);
        cudaEventDestroy(d0[i]);
        cudaEventDestroy(d1[i]);
    }
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFree(d_win);
    cudaFree(d_y);
    cudaFree(d_h);
    cudaHostUnregister(y_out);
    cudaHostUnregister(packed);
    free(packed);
    free(e0);
    free(e1);
    free(k0);
    free(k1);
    free(c0);
    free(c1);
    free(d0);
    free(d1);
    return 0;
}

int path_mapped_measure(int buf, int fir, int sources, const float *x, const float *h_flip, int total_blocks,
                        int warmup, int measured, float *y_out, double *us_e2e, double *us_kernel,
                        double *us_copy, long *bytes_per_block) {
    const int window = fir - 1 + buf;
    const int use_const = coeffs_use_const(sources, fir);
    const size_t win_floats = static_cast<size_t>(sources) * static_cast<size_t>(window);
    const size_t y_floats = static_cast<size_t>(sources) * 2u * static_cast<size_t>(buf);
    const size_t win_bytes = static_cast<size_t>(total_blocks) * win_floats * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(measured) * y_floats * sizeof(float);
    float *packed = nullptr;
    float *mapped_y = nullptr;
    float *d_win = nullptr;
    float *d_y = nullptr;
    float *d_h = nullptr;
    cudaEvent_t ev0, ev1;
    if (posix_memalign(reinterpret_cast<void **>(&packed), 4096, win_bytes) != 0 ||
        posix_memalign(reinterpret_cast<void **>(&mapped_y), 4096, out_bytes) != 0) {
        return 1;
    }
    pack_windows(packed, x, buf, fir, sources, total_blocks);
    cudaError_t reg_x = cudaHostRegister(packed, win_bytes, cudaHostRegisterMapped);
    cudaError_t reg_y = cudaHostRegister(mapped_y, out_bytes, cudaHostRegisterMapped);
    if (reg_x != cudaSuccess || reg_y != cudaSuccess) {
        fprintf(stderr, "path mapped register: %s / %s\n", cudaGetErrorString(reg_x),
                cudaGetErrorString(reg_y));
        if (reg_x == cudaSuccess) {
            cudaHostUnregister(packed);
        }
        free(packed);
        free(mapped_y);
        return 2;
    }
    CUDA_CHECK(cudaHostGetDevicePointer(&d_win, packed, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_y, mapped_y, 0));
    if (!use_const) {
        CUDA_CHECK(cudaMalloc(&d_h, static_cast<size_t>(sources) * 2u * static_cast<size_t>(fir) * sizeof(float)));
    }
    if (upload_coeffs(h_flip, sources, fir, use_const, d_h)) {
        return 1;
    }
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    for (int b = 0; b < warmup; ++b) {
        launch_window(d_win + static_cast<size_t>(b) * win_floats, d_y, d_h, buf, fir, sources, use_const, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int b = warmup; b < total_blocks; ++b) {
        const int mi = b - warmup;
        CUDA_CHECK(cudaEventRecord(ev0));
        launch_window(d_win + static_cast<size_t>(b) * win_floats, d_y + static_cast<size_t>(mi) * y_floats, d_h,
                      buf, fir, sources, use_const, 0);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        us_e2e[mi] = static_cast<double>(ms) * 1000.0;
        us_kernel[mi] = us_e2e[mi];
        us_copy[mi] = 0.0;
    }
    CUDA_CHECK(cudaGetLastError());
    memcpy(y_out, mapped_y, out_bytes);
    *bytes_per_block = 0;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_h);
    cudaHostUnregister(packed);
    cudaHostUnregister(mapped_y);
    free(packed);
    free(mapped_y);
    return 0;
}

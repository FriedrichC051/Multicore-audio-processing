#include "e2e.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

enum { kE2eBlock = 512, kConstBytes = 60 * 1024 };

__constant__ float c_coeff[kConstBytes / sizeof(float)];

static int coeffs_use_const(int sources, int fir) {
    size_t bytes = (size_t)sources * 2u * (size_t)fir * sizeof(float);
    return bytes <= (size_t)kConstBytes;
}

template <int UseConst>
__global__ void fir_e2e_kernel(const float *__restrict__ hist_in, const float *__restrict__ neu,
                               float *__restrict__ hist_out, float *__restrict__ y,
                               const float *__restrict__ hglob, int buf, int fir) {
    const int ch = (int)blockIdx.x;
    const int src = (int)blockIdx.y;
    const int hist = fir - 1;
    extern __shared__ float sx[];

    for (int i = threadIdx.x; i < hist; i += blockDim.x) {
        sx[i] = hist_in[src * hist + i];
    }
    for (int i = threadIdx.x; i < buf; i += blockDim.x) {
        sx[hist + i] = neu[src * buf + i];
    }
    __syncthreads();

    const float *hc =
        UseConst ? (c_coeff + ((src * 2 + ch) * fir)) : (hglob + ((src * 2 + ch) * fir));
    float *yc = y + (src * 2 + ch) * buf;
    for (int j = threadIdx.x; j < buf; j += blockDim.x) {
        const float *x = sx + j;
        float acc = 0.0f;
        int k = 0;
#pragma unroll 1
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
    __syncthreads();
    if (ch == 0) {
        for (int i = threadIdx.x; i < hist; i += blockDim.x) {
            hist_out[src * hist + i] = sx[buf + i];
        }
    }
}

static int upload_static_coeffs(const float *h_flip, int sources, int fir, int use_const, float *d_h) {
    size_t bytes = (size_t)sources * 2u * (size_t)fir * sizeof(float);
    if (use_const) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_coeff, h_flip, bytes));
    } else {
        CUDA_CHECK(cudaMemcpy(d_h, h_flip, bytes, cudaMemcpyHostToDevice));
    }
    return 0;
}

int e2e_time_measure(int buf, int fir, int sources, int depth, int swap, int norient, const float *x,
                     const float *h_flip, int total_blocks, int warmup, int measured, float *y_out,
                     double *us_e2e, double *us_kernel) {
    const int hist = fir - 1;
    const int use_const = coeffs_use_const(sources, fir);
    const size_t coeff_floats = (size_t)sources * 2u * (size_t)fir;
    const size_t new_floats = (size_t)sources * (size_t)buf;
    const size_t y_floats = (size_t)sources * 2u * (size_t)buf;
    const int smem = (hist + buf) * (int)sizeof(float);
    float *d_hist[2] = {NULL, NULL};
    float *d_new = NULL;
    float *d_y = NULL;
    float *d_h = NULL;
    float *pack = NULL;
    cudaStream_t streams[3];
    cudaEvent_t *e2e0 = NULL;
    cudaEvent_t *e2e1 = NULL;
    cudaEvent_t *k0 = NULL;
    cudaEvent_t *k1 = NULL;
    cudaEvent_t *hist_ev = NULL;
    int b, s, i;

    if (depth < 1 || depth > 3 || warmup < 0 || measured < 1 || total_blocks < warmup + measured) {
        fprintf(stderr, "e2e_time_measure: bad shape\n");
        return 1;
    }
    if ((size_t)smem > 48u * 1024u) {
        fprintf(stderr, "e2e shared window %d bytes exceeds the default limit\n", smem);
        return 1;
    }

    CUDA_CHECK(cudaMalloc(&d_hist[0], (size_t)sources * (size_t)hist * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hist[1], (size_t)sources * (size_t)hist * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new, (size_t)depth * new_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)depth * y_floats * sizeof(float)));
    if (!use_const) {
        CUDA_CHECK(cudaMalloc(&d_h, coeff_floats * sizeof(float)));
    }
    CUDA_CHECK(cudaHostAlloc(&pack, (size_t)total_blocks * new_floats * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaMemset(d_hist[0], 0, (size_t)sources * (size_t)hist * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hist[1], 0, (size_t)sources * (size_t)hist * sizeof(float)));

    for (b = 0; b < total_blocks; ++b) {
        for (s = 0; s < sources; ++s) {
            const float *src = x + ((size_t)s * (size_t)(hist + total_blocks * buf)) + (size_t)hist +
                               (size_t)b * (size_t)buf;
            memcpy(pack + ((size_t)b * sources + (size_t)s) * (size_t)buf, src, (size_t)buf * sizeof(float));
        }
    }

    e2e0 = (cudaEvent_t *)calloc((size_t)measured, sizeof(cudaEvent_t));
    e2e1 = (cudaEvent_t *)calloc((size_t)measured, sizeof(cudaEvent_t));
    k0 = (cudaEvent_t *)calloc((size_t)measured, sizeof(cudaEvent_t));
    k1 = (cudaEvent_t *)calloc((size_t)measured, sizeof(cudaEvent_t));
    hist_ev = (cudaEvent_t *)calloc((size_t)total_blocks, sizeof(cudaEvent_t));
    if (!e2e0 || !e2e1 || !k0 || !k1 || !hist_ev) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (i = 0; i < measured; ++i) {
        CUDA_CHECK(cudaEventCreate(&e2e0[i]));
        CUDA_CHECK(cudaEventCreate(&e2e1[i]));
        CUDA_CHECK(cudaEventCreate(&k0[i]));
        CUDA_CHECK(cudaEventCreate(&k1[i]));
    }
    for (b = 0; b < total_blocks; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&hist_ev[b], cudaEventDisableTiming));
    }
    for (i = 0; i < depth; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    if (!swap) {
        if (upload_static_coeffs(h_flip, sources, fir, use_const, d_h)) {
            return 1;
        }
    }

    for (b = 0; b < total_blocks; ++b) {
        const int slot = b % depth;
        const int rec = b >= warmup;
        const int mi = b - warmup;
        cudaStream_t st = streams[slot];
        if (rec) {
            CUDA_CHECK(cudaEventRecord(e2e0[mi], st));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_new + (size_t)slot * new_floats,
                                   pack + (size_t)b * new_floats, new_floats * sizeof(float),
                                   cudaMemcpyHostToDevice, st));
        if (swap) {
            const float *h = h_flip + (size_t)(b % norient) * coeff_floats;
            if (use_const) {
                CUDA_CHECK(cudaMemcpyToSymbolAsync(c_coeff, h, coeff_floats * sizeof(float), 0,
                                                   cudaMemcpyHostToDevice, st));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_h, h, coeff_floats * sizeof(float), cudaMemcpyHostToDevice, st));
            }
        }
        if (b > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(st, hist_ev[b - 1], 0));
        }
        if (rec) {
            CUDA_CHECK(cudaEventRecord(k0[mi], st));
        }
        if (use_const) {
            fir_e2e_kernel<1><<<dim3(2, sources), kE2eBlock, (size_t)smem, st>>>(
                d_hist[b % 2], d_new + (size_t)slot * new_floats, d_hist[(b + 1) % 2],
                d_y + (size_t)slot * y_floats, d_h, buf, fir);
        } else {
            fir_e2e_kernel<0><<<dim3(2, sources), kE2eBlock, (size_t)smem, st>>>(
                d_hist[b % 2], d_new + (size_t)slot * new_floats, d_hist[(b + 1) % 2],
                d_y + (size_t)slot * y_floats, d_h, buf, fir);
        }
        if (rec) {
            CUDA_CHECK(cudaEventRecord(k1[mi], st));
        }
        CUDA_CHECK(cudaEventRecord(hist_ev[b], st));
        if (rec) {
            CUDA_CHECK(cudaMemcpyAsync(y_out + (size_t)mi * y_floats, d_y + (size_t)slot * y_floats,
                                       y_floats * sizeof(float), cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaEventRecord(e2e1[mi], st));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    for (i = 0; i < measured; ++i) {
        float e2e_ms = 0.0f;
        float ker_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&e2e_ms, e2e0[i], e2e1[i]));
        CUDA_CHECK(cudaEventElapsedTime(&ker_ms, k0[i], k1[i]));
        us_e2e[i] = (double)e2e_ms * 1000.0;
        us_kernel[i] = (double)ker_ms * 1000.0;
    }

    for (i = 0; i < measured; ++i) {
        cudaEventDestroy(e2e0[i]);
        cudaEventDestroy(e2e1[i]);
        cudaEventDestroy(k0[i]);
        cudaEventDestroy(k1[i]);
    }
    for (b = 0; b < total_blocks; ++b) {
        cudaEventDestroy(hist_ev[b]);
    }
    for (i = 0; i < depth; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_hist[0]);
    cudaFree(d_hist[1]);
    cudaFree(d_new);
    cudaFree(d_y);
    cudaFree(d_h);
    cudaFreeHost(pack);
    free(e2e0);
    free(e2e1);
    free(k0);
    free(k1);
    free(hist_ev);
    return 0;
}

#include "e2e.h"

#include <cuda_runtime.h>
#include <cufft.h>

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

#define CUFFT_CHECK(expr)                                                            \
    do {                                                                             \
        cufftResult _r = (expr);                                                     \
        if (_r != CUFFT_SUCCESS) {                                                   \
            fprintf(stderr, "%s:%d cufft %d\n", __FILE__, __LINE__, (int)_r);       \
            return 1;                                                                \
        }                                                                            \
    } while (0)

enum { kPlanCap = 8 };

typedef struct {
    int n;
    cufftHandle r2c;
    cufftHandle c2r;
} FftPlan;

static FftPlan g_plans[kPlanCap];
static int g_nplans = 0;

static int next_pow2(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

static FftPlan *plan_for(int n, cudaStream_t stream) {
    int i;
    for (i = 0; i < g_nplans; ++i) {
        if (g_plans[i].n == n) {
            if (cufftSetStream(g_plans[i].r2c, stream) != CUFFT_SUCCESS ||
                cufftSetStream(g_plans[i].c2r, stream) != CUFFT_SUCCESS) {
                return NULL;
            }
            return &g_plans[i];
        }
    }
    if (g_nplans >= kPlanCap) {
        fprintf(stderr, "too many fft plans\n");
        return NULL;
    }
    g_plans[g_nplans].n = n;
    if (cufftPlan1d(&g_plans[g_nplans].r2c, n, CUFFT_R2C, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan r2c failed\n");
        return NULL;
    }
    if (cufftPlan1d(&g_plans[g_nplans].c2r, n, CUFFT_C2R, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan c2r failed\n");
        return NULL;
    }
    cufftSetStream(g_plans[g_nplans].r2c, stream);
    cufftSetStream(g_plans[g_nplans].c2r, stream);
    return &g_plans[g_nplans++];
}

void e2e_fft_shutdown(void) {
    int i;
    for (i = 0; i < g_nplans; ++i) {
        cufftDestroy(g_plans[i].r2c);
        cufftDestroy(g_plans[i].c2r);
    }
    g_nplans = 0;
}

__global__ void scale_spectrum(cufftComplex *spec, int nfreq, float scale) {
    int i = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (i < nfreq) {
        spec[i].x *= scale;
        spec[i].y *= scale;
    }
}

__global__ void mul_spectrum(cufftComplex *spec, const cufftComplex *h, int nfreq) {
    int i = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (i < nfreq) {
        float ax = spec[i].x;
        float ay = spec[i].y;
        float bx = h[i].x;
        float by = h[i].y;
        spec[i].x = ax * bx - ay * by;
        spec[i].y = ax * by + ay * bx;
    }
}

__global__ void ola_stride_kernel(const float *time, float *tail, float *y, int nfft, int hop, int hist) {
    int ch = (int)blockIdx.x;
    const float *t = time + (size_t)ch * (size_t)nfft;
    float *tl = tail + (size_t)ch * (size_t)hist;
    float *yc = y + (size_t)ch * (size_t)hop;
    for (int i = (int)threadIdx.x; i < hop; i += (int)blockDim.x) {
        float v = t[i];
        if (i < hist) {
            v += tl[i];
        }
        yc[i] = v;
    }
    __syncthreads();
    for (int i = (int)threadIdx.x; i < hist; i += (int)blockDim.x) {
        tl[i] = t[hop + i];
    }
}

int e2e_fft_measure(int buf, int fir, int k, const float *x, const float *h_orig, int total_blocks,
                    int warmup, int measured, float *y_out, double *us_e2e, double *us_compute,
                    int *groups_out) {
    const int hist = fir - 1;
    const int hop = k * buf;
    const int nfft = next_pow2(hop + fir - 1);
    const int nfreq = nfft / 2 + 1;
    const int warmup_groups = warmup / k;
    const int measured_groups = measured / k;
    const int total_groups = total_blocks / k;
    float *d_time = NULL;
    cufftComplex *d_spec = NULL;
    cufftComplex *d_hspec = NULL;
    float *d_tail = NULL;
    float *d_y = NULL;
    float *hpad = NULL;
    float *pack = NULL;
    cudaStream_t stream = NULL;
    cudaEvent_t *e0 = NULL;
    cudaEvent_t *e1 = NULL;
    cudaEvent_t *c0 = NULL;
    cudaEvent_t *c1 = NULL;
    FftPlan *plan = NULL;
    int g, ch, i;

    if (k < 1 || warmup % k != 0 || measured % k != 0 || total_blocks < warmup + measured) {
        fprintf(stderr, "fft grouping does not divide the block counts\n");
        return 1;
    }
    *groups_out = measured_groups;

    CUDA_CHECK(cudaStreamCreate(&stream));
    plan = plan_for(nfft, stream);
    if (!plan) {
        return 1;
    }
    CUDA_CHECK(cudaMalloc(&d_time, (size_t)2 * (size_t)nfft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spec, (size_t)2 * (size_t)nfreq * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_hspec, (size_t)2 * (size_t)nfreq * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_tail, (size_t)2 * (size_t)hist * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)2 * (size_t)hop * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_tail, 0, (size_t)2 * (size_t)hist * sizeof(float)));
    CUDA_CHECK(cudaHostAlloc(&pack, (size_t)total_groups * (size_t)hop * sizeof(float), cudaHostAllocDefault));

    hpad = (float *)calloc((size_t)nfft, sizeof(float));
    if (!hpad) {
        return 1;
    }
    for (ch = 0; ch < 2; ++ch) {
        memset(hpad, 0, (size_t)nfft * sizeof(float));
        memcpy(hpad, h_orig + (size_t)ch * (size_t)fir, (size_t)fir * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_time, hpad, (size_t)nfft * sizeof(float), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecR2C(plan->r2c, d_time, d_hspec + (size_t)ch * (size_t)nfreq));
        scale_spectrum<<<(nfreq + 255) / 256, 256, 0, stream>>>(d_hspec + (size_t)ch * (size_t)nfreq, nfreq,
                                                               1.0f / (float)nfft);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (g = 0; g < total_groups; ++g) {
        const float *src = x + (size_t)hist + (size_t)g * (size_t)hop;
        memcpy(pack + (size_t)g * (size_t)hop, src, (size_t)hop * sizeof(float));
    }

    e0 = (cudaEvent_t *)calloc((size_t)measured_groups, sizeof(cudaEvent_t));
    e1 = (cudaEvent_t *)calloc((size_t)measured_groups, sizeof(cudaEvent_t));
    c0 = (cudaEvent_t *)calloc((size_t)measured_groups, sizeof(cudaEvent_t));
    c1 = (cudaEvent_t *)calloc((size_t)measured_groups, sizeof(cudaEvent_t));
    if (!e0 || !e1 || !c0 || !c1) {
        return 1;
    }
    for (i = 0; i < measured_groups; ++i) {
        CUDA_CHECK(cudaEventCreate(&e0[i]));
        CUDA_CHECK(cudaEventCreate(&e1[i]));
        CUDA_CHECK(cudaEventCreate(&c0[i]));
        CUDA_CHECK(cudaEventCreate(&c1[i]));
    }

    for (g = 0; g < total_groups; ++g) {
        const int rec = g >= warmup_groups;
        const int mi = g - warmup_groups;
        if (rec) {
            CUDA_CHECK(cudaEventRecord(e0[mi], stream));
        }
        CUDA_CHECK(cudaMemsetAsync(d_time, 0, (size_t)2 * (size_t)nfft * sizeof(float), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_time, pack + (size_t)g * (size_t)hop, (size_t)hop * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_time + nfft, pack + (size_t)g * (size_t)hop, (size_t)hop * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        if (rec) {
            CUDA_CHECK(cudaEventRecord(c0[mi], stream));
        }
        for (ch = 0; ch < 2; ++ch) {
            CUFFT_CHECK(cufftExecR2C(plan->r2c, d_time + (size_t)ch * (size_t)nfft,
                                     d_spec + (size_t)ch * (size_t)nfreq));
            mul_spectrum<<<(nfreq + 255) / 256, 256, 0, stream>>>(
                d_spec + (size_t)ch * (size_t)nfreq, d_hspec + (size_t)ch * (size_t)nfreq, nfreq);
            CUFFT_CHECK(cufftExecC2R(plan->c2r, d_spec + (size_t)ch * (size_t)nfreq,
                                     d_time + (size_t)ch * (size_t)nfft));
        }
        ola_stride_kernel<<<2, 256, 0, stream>>>(d_time, d_tail, d_y, nfft, hop, hist);
        if (rec) {
            CUDA_CHECK(cudaEventRecord(c1[mi], stream));
            for (i = 0; i < k; ++i) {
                float *dst = y_out + ((size_t)mi * (size_t)k + (size_t)i) * 2u * (size_t)buf;
                CUDA_CHECK(cudaMemcpyAsync(dst, d_y + (size_t)i * (size_t)buf, (size_t)buf * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(dst + buf, d_y + (size_t)hop + (size_t)i * (size_t)buf,
                                           (size_t)buf * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaEventRecord(e1[mi], stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    for (i = 0; i < measured_groups; ++i) {
        float e_ms = 0.0f;
        float c_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&e_ms, e0[i], e1[i]));
        CUDA_CHECK(cudaEventElapsedTime(&c_ms, c0[i], c1[i]));
        us_e2e[i] = (double)e_ms * 1000.0;
        us_compute[i] = (double)c_ms * 1000.0;
    }

    for (i = 0; i < measured_groups; ++i) {
        cudaEventDestroy(e0[i]);
        cudaEventDestroy(e1[i]);
        cudaEventDestroy(c0[i]);
        cudaEventDestroy(c1[i]);
    }
    cudaStreamDestroy(stream);
    cudaFree(d_time);
    cudaFree(d_spec);
    cudaFree(d_hspec);
    cudaFree(d_tail);
    cudaFree(d_y);
    cudaFreeHost(pack);
    free(hpad);
    free(e0);
    free(e1);
    free(c0);
    free(c1);
    return 0;
}

#include "conv.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

__constant__ float c_hflip[2 * CONV_FIR];

static int grid_block(int block_size) {
    if (block_size < 32) {
        return 32;
    }
    if (block_size > 1024) {
        return 1024;
    }
    return block_size;
}

__global__ void fir_f32_naive_kernel(const float *__restrict__ ext, const float *__restrict__ h,
                                     float *__restrict__ y) {
    const int ch = blockIdx.x;
    const float *hc = h + ch * CONV_FIR;
    float *yc = y + ch * CONV_BUF;
    for (int j = threadIdx.x; j < CONV_BUF; j += blockDim.x) {
        const float *x = ext + j;
        float acc = 0.0f;
        for (int k = 0; k < CONV_FIR; ++k) {
            acc += x[k] * hc[k];
        }
        yc[j] = acc;
    }
}

__global__ void fir_f32_const_kernel(const float *__restrict__ ext, float *__restrict__ y) {
    const int ch = blockIdx.x;
    const float *hc = c_hflip + ch * CONV_FIR;
    float *yc = y + ch * CONV_BUF;
    for (int j = threadIdx.x; j < CONV_BUF; j += blockDim.x) {
        const float *x = ext + j;
        float acc = 0.0f;
        for (int k = 0; k < CONV_FIR; ++k) {
            acc += x[k] * hc[k];
        }
        yc[j] = acc;
    }
}

__global__ void fir_f32_shared_kernel(const float *__restrict__ ext, const float *__restrict__ h,
                                      float *__restrict__ y) {
    __shared__ float sx[CONV_HIST + CONV_BUF];
    const int ch = blockIdx.x;
    for (int i = threadIdx.x; i < CONV_HIST + CONV_BUF; i += blockDim.x) {
        sx[i] = ext[i];
    }
    __syncthreads();

    const float *hc = h + ch * CONV_FIR;
    float *yc = y + ch * CONV_BUF;
    for (int j = threadIdx.x; j < CONV_BUF; j += blockDim.x) {
        const float *x = sx + j;
        float acc = 0.0f;
        for (int k = 0; k < CONV_FIR; ++k) {
            acc += x[k] * hc[k];
        }
        yc[j] = acc;
    }
}

__global__ void fir_f32_unroll_kernel(const float *__restrict__ ext, float *__restrict__ y) {
    __shared__ float sx[CONV_HIST + CONV_BUF];
    const int ch = blockIdx.x;
    for (int i = threadIdx.x; i < CONV_HIST + CONV_BUF; i += blockDim.x) {
        sx[i] = ext[i];
    }
    __syncthreads();

    const float *hc = c_hflip + ch * CONV_FIR;
    float *yc = y + ch * CONV_BUF;
    for (int j = threadIdx.x; j < CONV_BUF; j += blockDim.x) {
        const float *x = sx + j;
        float acc = 0.0f;
#pragma unroll 16
        for (int k = 0; k < CONV_FIR; ++k) {
            acc += x[k] * hc[k];
        }
        yc[j] = acc;
    }
}

extern "C" void fir_f32_const_upload(const float *h_host) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_hflip, h_host, sizeof(c_hflip)));
}

extern "C" void fir_f32_naive_launch(const float *d_x, const float *d_h, float *d_y, int abs_index,
                                     int block_size) {
    const float *ext = d_x + abs_index * CONV_BUF;
    fir_f32_naive_kernel<<<2, grid_block(block_size)>>>(ext, d_h, d_y);
}

extern "C" void fir_f32_const_launch(const float *d_x, float *d_y, int abs_index, int block_size) {
    const float *ext = d_x + abs_index * CONV_BUF;
    fir_f32_const_kernel<<<2, grid_block(block_size)>>>(ext, d_y);
}

extern "C" void fir_f32_shared_launch(const float *d_x, const float *d_h, float *d_y, int abs_index,
                                      int block_size) {
    const float *ext = d_x + abs_index * CONV_BUF;
    fir_f32_shared_kernel<<<2, grid_block(block_size)>>>(ext, d_h, d_y);
}

extern "C" void fir_f32_unroll_launch(const float *d_x, float *d_y, int abs_index, int block_size) {
    const float *ext = d_x + abs_index * CONV_BUF;
    fir_f32_unroll_kernel<<<2, grid_block(block_size)>>>(ext, d_y);
}

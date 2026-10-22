#include "conv.h"

#include <cuda_runtime.h>

__device__ int pack_i8x4(const int8_t *p) {
    return (int)(uint8_t)p[0] | ((int)(uint8_t)p[1] << 8) | ((int)(uint8_t)p[2] << 16) |
           ((int)(uint8_t)p[3] << 24);
}

__global__ void fir_i8_dp4a_kernel(const int8_t *__restrict__ ext, const int8_t *__restrict__ h,
                                   float *__restrict__ y) {
    const int ch = blockIdx.x;
    const int8_t *hc = h + ch * CONV_FIR;
    float *yc = y + ch * CONV_BUF;
    const float scale = 1.0f / ((float)CONV_I8_SCALE * (float)CONV_I8_SCALE);
    for (int j = threadIdx.x; j < CONV_BUF; j += blockDim.x) {
        const int8_t *x = ext + j;
        int acc = 0;
        for (int k = 0; k < CONV_FIR; k += 4) {
            acc = __dp4a(pack_i8x4(x + k), pack_i8x4(hc + k), acc);
        }
        yc[j] = (float)acc * scale;
    }
}

extern "C" void fir_i8_dp4a_launch(const int8_t *d_x, const int8_t *d_h, float *d_y, int abs_index) {
    const int8_t *ext = d_x + abs_index * CONV_BUF;
    fir_i8_dp4a_kernel<<<2, 256>>>(ext, d_h, d_y);
}

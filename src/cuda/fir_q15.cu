#include "conv.h"

#include <cuda_runtime.h>

__global__ void fir_q15_i64_kernel(const int16_t *__restrict__ ext, const int16_t *__restrict__ h,
                                   int16_t *__restrict__ y) {
    const int ch = blockIdx.x;
    const int16_t *hc = h + ch * CONV_FIR;
    int16_t *yc = y + ch * CONV_BUF;
    for (int j = threadIdx.x; j < CONV_BUF; j += blockDim.x) {
        const int16_t *x = ext + j;
        int64_t acc = 0;
        for (int k = 0; k < CONV_FIR; ++k) {
            acc += (int64_t)x[k] * (int64_t)hc[k];
        }
        int64_t q = acc >> 15;
        if (q > 32767) {
            q = 32767;
        } else if (q < -32768) {
            q = -32768;
        }
        yc[j] = (int16_t)q;
    }
}

extern "C" void fir_q15_i64_launch(const int16_t *d_x, const int16_t *d_h, int16_t *d_y,
                                   int abs_index) {
    const int16_t *ext = d_x + abs_index * CONV_BUF;
    fir_q15_i64_kernel<<<2, 256>>>(ext, d_h, d_y);
}

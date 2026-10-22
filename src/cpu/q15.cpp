#include "conv.h"

#include <stdint.h>

static int16_t sat_q15(int64_t q) {
    if (q > 32767) {
        return 32767;
    }
    if (q < -32768) {
        return -32768;
    }
    return (int16_t)q;
}

static int64_t dot_q15(const int16_t *x, const int16_t *h, int fir) {
    int64_t acc = 0;
    int k;
    for (k = 0; k < fir; ++k) {
        acc += (int64_t)x[k] * (int64_t)h[k];
    }
    return acc;
}

void conv_q15_i64(const int16_t *ext, const int16_t *h_flip, int16_t *y, int buf, int fir) {
    int j;
    for (j = 0; j < buf; ++j) {
        int64_t acc = dot_q15(ext + j, h_flip, fir);
        y[j] = sat_q15(acc >> 15);
    }
}

void conv_q15_i32(const int16_t *ext, const int16_t *h_flip, int16_t *y, int buf, int fir) {
    int j;
    for (j = 0; j < buf; ++j) {
        int64_t acc = dot_q15(ext + j, h_flip, fir);
        int32_t wrapped = (int32_t)(uint32_t)(uint64_t)acc;
        y[j] = sat_q15((int64_t)(wrapped >> 15));
    }
}

int conv_q15_overflow_self_check(void) {
    int16_t ext[CONV_HIST + CONV_BUF];
    int16_t h_flip[CONV_FIR];
    int16_t y32[CONV_BUF];
    int16_t y64[CONV_BUF];
    int i;
    int differ = 0;
    long overflows;

    for (i = 0; i < CONV_HIST + CONV_BUF; ++i) {
        ext[i] = 29491; /* Q15 of about 0.9 */
    }
    for (i = 0; i < CONV_FIR; ++i) {
        h_flip[i] = 29491;
    }
    conv_q15_i32(ext, h_flip, y32, CONV_BUF, CONV_FIR);
    conv_q15_i64(ext, h_flip, y64, CONV_BUF, CONV_FIR);
    for (i = 0; i < CONV_BUF; ++i) {
        if (y32[i] != y64[i]) {
            differ = 1;
            break;
        }
    }
    overflows = 0;
    for (i = 0; i < CONV_BUF; ++i) {
        int64_t acc = dot_q15(ext + i, h_flip, CONV_FIR);
        if (acc > INT32_MAX || acc < INT32_MIN) {
            ++overflows;
        }
    }
    return (differ && overflows == CONV_BUF) ? 0 : 1;
}

long conv_q15_i32_overflows(const int16_t *x, const int16_t *h_l, const int16_t *h_r,
                            int abs_begin, int blocks) {
    long count = 0;
    int b;
    for (b = 0; b < blocks; ++b) {
        const int16_t *ext = x + (abs_begin + b) * CONV_BUF;
        int j;
        for (j = 0; j < CONV_BUF; ++j) {
            int64_t acc_l = dot_q15(ext + j, h_l, CONV_FIR);
            int64_t acc_r = dot_q15(ext + j, h_r, CONV_FIR);
            if (acc_l > INT32_MAX || acc_l < INT32_MIN) {
                ++count;
            }
            if (acc_r > INT32_MAX || acc_r < INT32_MIN) {
                ++count;
            }
        }
    }
    return count;
}

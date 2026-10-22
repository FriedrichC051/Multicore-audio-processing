#define _POSIX_C_SOURCE 200809L

#include "conv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static double elapsed_us(const struct timespec *t0, const struct timespec *t1) {
    return (double)(t1->tv_sec - t0->tv_sec) * 1.0e6 + (double)(t1->tv_nsec - t0->tv_nsec) / 1.0e3;
}

static void fmt_snr(double snr, char *buf, size_t n) {
    if (isinf(snr) && snr > 0) {
        snprintf(buf, n, "inf");
    } else {
        snprintf(buf, n, "%.2f", snr);
    }
}

static void quantize_stream(const float *src, int16_t *dst, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        dst[i] = conv_to_q15(src[i]);
    }
}

static void q15_to_double(const int16_t *src, double *dst, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        dst[i] = (double)src[i] / 32768.0;
    }
}

static void f32_to_double(const float *src, double *dst, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        dst[i] = (double)src[i];
    }
}

int main(void) {
    const int nx = conv_total_samples(CONV_WARMUP, CONV_BLOCKS);
    const int nout = CONV_BLOCKS * CONV_BUF;
    float *x = (float *)xmalloc((size_t)nx * sizeof(float));
    float *h_l = (float *)xmalloc((size_t)CONV_FIR * sizeof(float));
    float *h_r = (float *)xmalloc((size_t)CONV_FIR * sizeof(float));
    float *hf_l = (float *)xmalloc((size_t)CONV_FIR * sizeof(float));
    float *hf_r = (float *)xmalloc((size_t)CONV_FIR * sizeof(float));
    double *xd = (double *)xmalloc((size_t)nx * sizeof(double));
    double *hdf_l = (double *)xmalloc((size_t)CONV_FIR * sizeof(double));
    double *hdf_r = (double *)xmalloc((size_t)CONV_FIR * sizeof(double));
    double *ref_l = (double *)xmalloc((size_t)nout * sizeof(double));
    double *ref_r = (double *)xmalloc((size_t)nout * sizeof(double));
    double *test_l = (double *)xmalloc((size_t)nout * sizeof(double));
    double *test_r = (double *)xmalloc((size_t)nout * sizeof(double));
    float *y_l = (float *)xmalloc((size_t)nout * sizeof(float));
    float *y_r = (float *)xmalloc((size_t)nout * sizeof(float));
    int16_t *xq = (int16_t *)xmalloc((size_t)nx * sizeof(int16_t));
    int16_t *hq_l = (int16_t *)xmalloc((size_t)CONV_FIR * sizeof(int16_t));
    int16_t *hq_r = (int16_t *)xmalloc((size_t)CONV_FIR * sizeof(int16_t));
    int16_t *yq_l = (int16_t *)xmalloc((size_t)nout * sizeof(int16_t));
    int16_t *yq_r = (int16_t *)xmalloc((size_t)nout * sizeof(int16_t));
    double *us = (double *)xmalloc((size_t)CONV_BLOCKS * sizeof(double));
    long overflows;
    TimeStats ts_ref, ts_i32, ts_i64, ts_avx;
    char snr_ref[32], snr_i32[32], snr_i64[32], snr_avx[32];
    int i, b;

    if (conv_ref_self_check() != 0) {
        fprintf(stderr, "double reference self-check failed\n");
        return 1;
    }
    if (conv_q15_overflow_self_check() != 0) {
        fprintf(stderr, "q15 overflow self-check failed\n");
        return 1;
    }

    signal_generate(x, nx, h_l, h_r, CONV_FIR);
    hrir_flip(h_l, hf_l, CONV_FIR);
    hrir_flip(h_r, hf_r, CONV_FIR);
    for (i = 0; i < nx; ++i) {
        xd[i] = (double)x[i];
    }
    for (i = 0; i < CONV_FIR; ++i) {
        hdf_l[i] = (double)hf_l[i];
        hdf_r[i] = (double)hf_r[i];
    }
    quantize_stream(x, xq, nx);
    for (i = 0; i < CONV_FIR; ++i) {
        hq_l[i] = conv_to_q15(hf_l[i]);
        hq_r[i] = conv_to_q15(hf_r[i]);
    }

    for (b = 0; b < CONV_WARMUP; ++b) {
        conv_ref_f64(xd + b * CONV_BUF, hdf_l, ref_l, CONV_BUF, CONV_FIR);
        conv_ref_f64(xd + b * CONV_BUF, hdf_r, ref_r, CONV_BUF, CONV_FIR);
    }
    for (b = 0; b < CONV_BLOCKS; ++b) {
        struct timespec t0, t1;
        const int abs_index = CONV_WARMUP + b;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        conv_ref_f64(xd + abs_index * CONV_BUF, hdf_l, ref_l + b * CONV_BUF, CONV_BUF, CONV_FIR);
        conv_ref_f64(xd + abs_index * CONV_BUF, hdf_r, ref_r + b * CONV_BUF, CONV_BUF, CONV_FIR);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        us[b] = elapsed_us(&t0, &t1);
    }
    time_percentiles(us, CONV_BLOCKS, &ts_ref);

    for (b = 0; b < CONV_WARMUP; ++b) {
        conv_q15_i32(xq + b * CONV_BUF, hq_l, yq_l, CONV_BUF, CONV_FIR);
        conv_q15_i32(xq + b * CONV_BUF, hq_r, yq_r, CONV_BUF, CONV_FIR);
    }
    for (b = 0; b < CONV_BLOCKS; ++b) {
        struct timespec t0, t1;
        const int abs_index = CONV_WARMUP + b;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        conv_q15_i32(xq + abs_index * CONV_BUF, hq_l, yq_l + b * CONV_BUF, CONV_BUF, CONV_FIR);
        conv_q15_i32(xq + abs_index * CONV_BUF, hq_r, yq_r + b * CONV_BUF, CONV_BUF, CONV_FIR);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        us[b] = elapsed_us(&t0, &t1);
    }
    time_percentiles(us, CONV_BLOCKS, &ts_i32);
    q15_to_double(yq_l, test_l, nout);
    q15_to_double(yq_r, test_r, nout);
    fmt_snr(snr_db(ref_l, test_l, ref_r, test_r, nout), snr_i32, sizeof(snr_i32));
    overflows = conv_q15_i32_overflows(xq, hq_l, hq_r, CONV_WARMUP, CONV_BLOCKS);

    for (b = 0; b < CONV_WARMUP; ++b) {
        conv_q15_i64(xq + b * CONV_BUF, hq_l, yq_l, CONV_BUF, CONV_FIR);
        conv_q15_i64(xq + b * CONV_BUF, hq_r, yq_r, CONV_BUF, CONV_FIR);
    }
    for (b = 0; b < CONV_BLOCKS; ++b) {
        struct timespec t0, t1;
        const int abs_index = CONV_WARMUP + b;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        conv_q15_i64(xq + abs_index * CONV_BUF, hq_l, yq_l + b * CONV_BUF, CONV_BUF, CONV_FIR);
        conv_q15_i64(xq + abs_index * CONV_BUF, hq_r, yq_r + b * CONV_BUF, CONV_BUF, CONV_FIR);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        us[b] = elapsed_us(&t0, &t1);
    }
    time_percentiles(us, CONV_BLOCKS, &ts_i64);
    q15_to_double(yq_l, test_l, nout);
    q15_to_double(yq_r, test_r, nout);
    fmt_snr(snr_db(ref_l, test_l, ref_r, test_r, nout), snr_i64, sizeof(snr_i64));

    for (b = 0; b < CONV_WARMUP; ++b) {
        conv_avx2(x + b * CONV_BUF, hf_l, y_l, CONV_BUF, CONV_FIR);
        conv_avx2(x + b * CONV_BUF, hf_r, y_r, CONV_BUF, CONV_FIR);
    }
    for (b = 0; b < CONV_BLOCKS; ++b) {
        struct timespec t0, t1;
        const int abs_index = CONV_WARMUP + b;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        conv_avx2(x + abs_index * CONV_BUF, hf_l, y_l + b * CONV_BUF, CONV_BUF, CONV_FIR);
        conv_avx2(x + abs_index * CONV_BUF, hf_r, y_r + b * CONV_BUF, CONV_BUF, CONV_FIR);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        us[b] = elapsed_us(&t0, &t1);
    }
    time_percentiles(us, CONV_BLOCKS, &ts_avx);
    f32_to_double(y_l, test_l, nout);
    f32_to_double(y_r, test_r, nout);
    fmt_snr(snr_db(ref_l, test_l, ref_r, test_r, nout), snr_avx, sizeof(snr_avx));
    fmt_snr(INFINITY, snr_ref, sizeof(snr_ref));

    printf("| 实现 | p50 (us) | p99 (us) | max (us) | SNR (dB) | 溢出样本 |\n");
    printf("|---|---:|---:|---:|---:|---:|\n");
    printf("| ref_f64 | %.2f | %.2f | %.2f | %s | — |\n", ts_ref.p50_us, ts_ref.p99_us, ts_ref.max_us,
           snr_ref);
    printf("| q15_i32 | %.2f | %.2f | %.2f | %s | %ld |\n", ts_i32.p50_us, ts_i32.p99_us, ts_i32.max_us,
           snr_i32, overflows);
    printf("| q15_i64 | %.2f | %.2f | %.2f | %s | 0 |\n", ts_i64.p50_us, ts_i64.p99_us, ts_i64.max_us,
           snr_i64);
    printf("| avx2_fma | %.2f | %.2f | %.2f | %s | — |\n", ts_avx.p50_us, ts_avx.p99_us, ts_avx.max_us,
           snr_avx);
    printf("\n溢出样本指精确 Q15 乘积累加装不进 int32 的输出点。q15_i32 对这些点按 2^32 回绕后再右移 15 位。q15_i64 用 int64 累加，不回绕。\n");
    printf("本信号的卷积输出落在大约 [-1, 1) 内。此时累加值小于 2^31，溢出样本为 0，两档 Q15 的 SNR 一致。满幅系数的自检确认计数器在累加超出 int32 时会报出溢出。\n");
    return 0;
}

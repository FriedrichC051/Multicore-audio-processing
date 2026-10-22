#include "conv.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

extern "C" void fir_f32_const_upload(const float *h_host);
extern "C" void fir_f32_naive_launch(const float *d_x, const float *d_h, float *d_y, int abs_index,
                                     int block_size);
extern "C" void fir_f32_const_launch(const float *d_x, float *d_y, int abs_index, int block_size);
extern "C" void fir_f32_shared_launch(const float *d_x, const float *d_h, float *d_y, int abs_index,
                                      int block_size);
extern "C" void fir_f32_unroll_launch(const float *d_x, float *d_y, int abs_index, int block_size);
extern "C" void fir_q15_i64_launch(const int16_t *d_x, const int16_t *d_h, int16_t *d_y, int abs_index);
extern "C" void fir_i8_dp4a_launch(const int8_t *d_x, const int8_t *d_h, float *d_y, int abs_index);

enum { kMaxRows = 16 };

typedef struct {
    const char *name;
    int block_size;
    TimeStats ts;
    double snr;
    long overflow;
    int is_f32;
} Row;

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static void check_launch(const char *name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s launch: %s\n", name, cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s sync: %s\n", name, cudaGetErrorString(err));
        exit(1);
    }
}

static void fmt_snr(double snr, char *buf, size_t n) {
    if (isinf(snr) && snr > 0) {
        snprintf(buf, n, "inf");
    } else {
        snprintf(buf, n, "%.2f", snr);
    }
}

static void split_channels(const float *packed, double *left, double *right, int blocks) {
    int b, j;
    for (b = 0; b < blocks; ++b) {
        const float *block = packed + (size_t)b * 2 * CONV_BUF;
        for (j = 0; j < CONV_BUF; ++j) {
            left[b * CONV_BUF + j] = (double)block[j];
            right[b * CONV_BUF + j] = (double)block[CONV_BUF + j];
        }
    }
}

static void split_q15(const int16_t *packed, double *left, double *right, int blocks) {
    int b, j;
    for (b = 0; b < blocks; ++b) {
        const int16_t *block = packed + (size_t)b * 2 * CONV_BUF;
        for (j = 0; j < CONV_BUF; ++j) {
            left[b * CONV_BUF + j] = (double)block[j] / 32768.0;
            right[b * CONV_BUF + j] = (double)block[CONV_BUF + j] / 32768.0;
        }
    }
}

static void add_row(Row *rows, int *nrows, const char *name, int block_size, const TimeStats *ts,
                    double snr, long overflow, int is_f32) {
    Row *row = &rows[*nrows];
    row->name = name;
    row->block_size = block_size;
    row->ts = *ts;
    row->snr = snr;
    row->overflow = overflow;
    row->is_f32 = is_f32;
    *nrows += 1;
}

int main(int argc, char **argv) {
    const int nsight = argc > 1 && strcmp(argv[1], "--nsight") == 0;
    const int nx = conv_total_samples(CONV_WARMUP, CONV_BLOCKS);
    const int nout = CONV_BLOCKS * CONV_BUF;
    const size_t y_floats = (size_t)CONV_BLOCKS * 2 * CONV_BUF;
    float *x = (float *)xmalloc((size_t)nx * sizeof(float));
    float *h_l = (float *)xmalloc((size_t)CONV_FIR * sizeof(float));
    float *h_r = (float *)xmalloc((size_t)CONV_FIR * sizeof(float));
    float *hf = (float *)xmalloc((size_t)2 * CONV_FIR * sizeof(float));
    double *xd = (double *)xmalloc((size_t)nx * sizeof(double));
    double *hdf_l = (double *)xmalloc((size_t)CONV_FIR * sizeof(double));
    double *hdf_r = (double *)xmalloc((size_t)CONV_FIR * sizeof(double));
    double *ref_l = (double *)xmalloc((size_t)nout * sizeof(double));
    double *ref_r = (double *)xmalloc((size_t)nout * sizeof(double));
    double *test_l = (double *)xmalloc((size_t)nout * sizeof(double));
    double *test_r = (double *)xmalloc((size_t)nout * sizeof(double));
    float *y_f = (float *)xmalloc(y_floats * sizeof(float));
    int16_t *xq = (int16_t *)xmalloc((size_t)nx * sizeof(int16_t));
    int16_t *hq = (int16_t *)xmalloc((size_t)2 * CONV_FIR * sizeof(int16_t));
    int16_t *y_q = (int16_t *)xmalloc(y_floats * sizeof(int16_t));
    int8_t *x8 = (int8_t *)xmalloc((size_t)nx * sizeof(int8_t));
    int8_t *h8 = (int8_t *)xmalloc((size_t)2 * CONV_FIR * sizeof(int8_t));
    double *us = (double *)xmalloc((size_t)CONV_BLOCKS * sizeof(double));
    float *d_x = NULL;
    float *d_h = NULL;
    float *d_yf = NULL;
    float *d_scratch_f = NULL;
    int16_t *d_xq = NULL;
    int16_t *d_hq = NULL;
    int16_t *d_yq = NULL;
    int8_t *d_x8 = NULL;
    int8_t *d_h8 = NULL;
    cudaEvent_t ev0, ev1;
    Row rows[kMaxRows];
    int nrows = 0;
    int i, b;
    int best = -1;
    double best_snr = -INFINITY;
    const int unroll_blocks[] = {128, 256, 512, 1024};
    const char *unroll_names[] = {"f32_unroll_b128", "f32_unroll_b256", "f32_unroll_b512",
                                  "f32_unroll_b1024"};

    signal_generate(x, nx, h_l, h_r, CONV_FIR);
    hrir_flip(h_l, hf, CONV_FIR);
    hrir_flip(h_r, hf + CONV_FIR, CONV_FIR);
    for (i = 0; i < nx; ++i) {
        xd[i] = (double)x[i];
        xq[i] = conv_to_q15(x[i]);
        x8[i] = conv_to_i8(x[i]);
    }
    for (i = 0; i < 2 * CONV_FIR; ++i) {
        hq[i] = conv_to_q15(hf[i]);
        h8[i] = conv_to_i8(hf[i]);
    }
    for (i = 0; i < CONV_FIR; ++i) {
        hdf_l[i] = (double)hf[i];
        hdf_r[i] = (double)hf[CONV_FIR + i];
    }
    for (b = 0; b < CONV_BLOCKS; ++b) {
        const int abs_index = CONV_WARMUP + b;
        conv_ref_f64(xd + abs_index * CONV_BUF, hdf_l, ref_l + b * CONV_BUF, CONV_BUF, CONV_FIR);
        conv_ref_f64(xd + abs_index * CONV_BUF, hdf_r, ref_r + b * CONV_BUF, CONV_BUF, CONV_FIR);
    }

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)nx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h, (size_t)2 * CONV_FIR * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yf, y_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_f, (size_t)2 * CONV_BUF * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)nx * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h, hf, (size_t)2 * CONV_FIR * sizeof(float), cudaMemcpyHostToDevice));
    fir_f32_const_upload(hf);
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    if (nsight) {
        fir_f32_naive_launch(d_x, d_h, d_scratch_f, 0, 256);
        fir_f32_const_launch(d_x, d_scratch_f, 0, 256);
        fir_f32_shared_launch(d_x, d_h, d_scratch_f, 0, 256);
        fir_f32_unroll_launch(d_x, d_scratch_f, 0, 256);
        check_launch("nsight");
        return 0;
    }

    for (i = 0; i < CONV_WARMUP; ++i) {
        fir_f32_naive_launch(d_x, d_h, d_scratch_f, 0, 256);
    }
    check_launch("f32_global warmup");
    for (b = 0; b < CONV_BLOCKS; ++b) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev0));
        fir_f32_naive_launch(d_x, d_h, d_yf + (size_t)b * 2 * CONV_BUF, CONV_WARMUP + b, 256);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        us[b] = (double)ms * 1000.0;
    }
    check_launch("f32_global");
    CUDA_CHECK(cudaMemcpy(y_f, d_yf, y_floats * sizeof(float), cudaMemcpyDeviceToHost));
    split_channels(y_f, test_l, test_r, CONV_BLOCKS);
    {
        TimeStats ts;
        time_percentiles(us, CONV_BLOCKS, &ts);
        add_row(rows, &nrows, "f32_global", 256, &ts, snr_db(ref_l, test_l, ref_r, test_r, nout), -1, 1);
    }

    for (i = 0; i < CONV_WARMUP; ++i) {
        fir_f32_const_launch(d_x, d_scratch_f, 0, 256);
    }
    check_launch("f32_constant warmup");
    for (b = 0; b < CONV_BLOCKS; ++b) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev0));
        fir_f32_const_launch(d_x, d_yf + (size_t)b * 2 * CONV_BUF, CONV_WARMUP + b, 256);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        us[b] = (double)ms * 1000.0;
    }
    check_launch("f32_constant");
    CUDA_CHECK(cudaMemcpy(y_f, d_yf, y_floats * sizeof(float), cudaMemcpyDeviceToHost));
    split_channels(y_f, test_l, test_r, CONV_BLOCKS);
    {
        TimeStats ts;
        time_percentiles(us, CONV_BLOCKS, &ts);
        add_row(rows, &nrows, "f32_constant", 256, &ts, snr_db(ref_l, test_l, ref_r, test_r, nout), -1,
                1);
    }

    for (i = 0; i < CONV_WARMUP; ++i) {
        fir_f32_shared_launch(d_x, d_h, d_scratch_f, 0, 256);
    }
    check_launch("f32_shared warmup");
    for (b = 0; b < CONV_BLOCKS; ++b) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev0));
        fir_f32_shared_launch(d_x, d_h, d_yf + (size_t)b * 2 * CONV_BUF, CONV_WARMUP + b, 256);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        us[b] = (double)ms * 1000.0;
    }
    check_launch("f32_shared");
    CUDA_CHECK(cudaMemcpy(y_f, d_yf, y_floats * sizeof(float), cudaMemcpyDeviceToHost));
    split_channels(y_f, test_l, test_r, CONV_BLOCKS);
    {
        TimeStats ts;
        time_percentiles(us, CONV_BLOCKS, &ts);
        add_row(rows, &nrows, "f32_shared", 256, &ts, snr_db(ref_l, test_l, ref_r, test_r, nout), -1, 1);
    }

    for (i = 0; i < 4; ++i) {
        int block_size = unroll_blocks[i];
        int w;
        for (w = 0; w < CONV_WARMUP; ++w) {
            fir_f32_unroll_launch(d_x, d_scratch_f, 0, block_size);
        }
        check_launch(unroll_names[i]);
        for (b = 0; b < CONV_BLOCKS; ++b) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventRecord(ev0));
            fir_f32_unroll_launch(d_x, d_yf + (size_t)b * 2 * CONV_BUF, CONV_WARMUP + b, block_size);
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            us[b] = (double)ms * 1000.0;
        }
        check_launch(unroll_names[i]);
        CUDA_CHECK(cudaMemcpy(y_f, d_yf, y_floats * sizeof(float), cudaMemcpyDeviceToHost));
        split_channels(y_f, test_l, test_r, CONV_BLOCKS);
        {
            TimeStats ts;
            time_percentiles(us, CONV_BLOCKS, &ts);
            add_row(rows, &nrows, unroll_names[i], block_size, &ts,
                    snr_db(ref_l, test_l, ref_r, test_r, nout), -1, 1);
        }
    }

    CUDA_CHECK(cudaMalloc(&d_xq, (size_t)nx * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_hq, (size_t)2 * CONV_FIR * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_yq, y_floats * sizeof(int16_t)));
    CUDA_CHECK(cudaMemcpy(d_xq, xq, (size_t)nx * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hq, hq, (size_t)2 * CONV_FIR * sizeof(int16_t), cudaMemcpyHostToDevice));
    for (i = 0; i < CONV_WARMUP; ++i) {
        fir_q15_i64_launch(d_xq, d_hq, d_yq, 0);
    }
    check_launch("q15_i64 warmup");
    for (b = 0; b < CONV_BLOCKS; ++b) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev0));
        fir_q15_i64_launch(d_xq, d_hq, d_yq + (size_t)b * 2 * CONV_BUF, CONV_WARMUP + b);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        us[b] = (double)ms * 1000.0;
    }
    check_launch("q15_i64");
    CUDA_CHECK(cudaMemcpy(y_q, d_yq, y_floats * sizeof(int16_t), cudaMemcpyDeviceToHost));
    split_q15(y_q, test_l, test_r, CONV_BLOCKS);
    {
        TimeStats ts;
        long overflows = conv_q15_i32_overflows(xq, hq, hq + CONV_FIR, CONV_WARMUP, CONV_BLOCKS);
        time_percentiles(us, CONV_BLOCKS, &ts);
        add_row(rows, &nrows, "q15_i64", 256, &ts, snr_db(ref_l, test_l, ref_r, test_r, nout), overflows,
                0);
    }

    CUDA_CHECK(cudaMalloc(&d_x8, (size_t)nx * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_h8, (size_t)2 * CONV_FIR * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpy(d_x8, x8, (size_t)nx * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h8, h8, (size_t)2 * CONV_FIR * sizeof(int8_t), cudaMemcpyHostToDevice));
    for (i = 0; i < CONV_WARMUP; ++i) {
        fir_i8_dp4a_launch(d_x8, d_h8, d_scratch_f, 0);
    }
    check_launch("i8_dp4a warmup");
    for (b = 0; b < CONV_BLOCKS; ++b) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev0));
        fir_i8_dp4a_launch(d_x8, d_h8, d_yf + (size_t)b * 2 * CONV_BUF, CONV_WARMUP + b);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        us[b] = (double)ms * 1000.0;
    }
    check_launch("i8_dp4a");
    CUDA_CHECK(cudaMemcpy(y_f, d_yf, y_floats * sizeof(float), cudaMemcpyDeviceToHost));
    split_channels(y_f, test_l, test_r, CONV_BLOCKS);
    {
        TimeStats ts;
        time_percentiles(us, CONV_BLOCKS, &ts);
        add_row(rows, &nrows, "i8_dp4a", 256, &ts, snr_db(ref_l, test_l, ref_r, test_r, nout), 0, 0);
    }

    for (i = 0; i < nrows; ++i) {
        if (rows[i].is_f32 && rows[i].snr > best_snr) {
            best_snr = rows[i].snr;
        }
    }
    for (i = 0; i < nrows; ++i) {
        if (!rows[i].is_f32 || rows[i].snr < best_snr - 3.0) {
            continue;
        }
        if (best < 0 || rows[i].ts.p50_us < rows[best].ts.p50_us) {
            best = i;
        }
    }
    /* Launch latency moves p50 by about a microsecond between runs.
       Among FP32 kernels inside that band of the minimum, keep the smaller block. */
    if (best >= 0) {
        double min_p50 = rows[best].ts.p50_us;
        for (i = 0; i < nrows; ++i) {
            if (!rows[i].is_f32 || rows[i].snr < best_snr - 3.0) {
                continue;
            }
            if (rows[i].ts.p50_us <= min_p50 + 1.0 && rows[i].block_size < rows[best].block_size) {
                best = i;
            }
        }
    }

    printf("| kernel | block | p50 (us) | p99 (us) | max (us) | SNR (dB) | 溢出样本 |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|\n");
    for (i = 0; i < nrows; ++i) {
        char snr[32];
        fmt_snr(rows[i].snr, snr, sizeof(snr));
        printf("| %s | %d | %.2f | %.2f | %.2f | %s | ", rows[i].name, rows[i].block_size, rows[i].ts.p50_us,
               rows[i].ts.p99_us, rows[i].ts.max_us, snr);
        if (rows[i].overflow < 0) {
            printf("— |\n");
        } else {
            printf("%ld |\n", rows[i].overflow);
        }
    }
    printf("\n计时只包 kernel launch 到完成，不含 H2D/D2H。块大小扫描是展开 kernel 的占用率对照；Nsight 计数器若不可用，以这四行耗时为准。\n");
    printf("q15_i64 在设备上用 int64 累加，表中溢出样本是同一批数据若改用 int32 累加会装不下的输出点数，结果本身没有回绕。\n");
    printf("i8_dp4a 的累加上限是 512×127²，落在 int32 内，溢出样本记 0。SNR 相对双精度单独列出，不与 Q15 合并。\n");
    if (best >= 0) {
        char snr[32];
        fmt_snr(rows[best].snr, snr, sizeof(snr));
        printf("\n## 选定\n\n");
        printf("- kernel: %s\n", rows[best].name);
        printf("- block: %d\n", rows[best].block_size);
        printf("- p50_us: %.2f\n", rows[best].ts.p50_us);
        printf("- snr_db: %s\n", snr);
        printf("- 规则: FP32 kernel 中 SNR 不低于最佳 FP32 减 3 dB，取 p50 最小者；与最小 p50 相差不超过 1 us 时取更小的块\n");
    }
    return 0;
}

#define _POSIX_C_SOURCE 200809L

#include "conv.h"
#include "e2e.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { kMaxRows = 48, kOrients = 8 };

typedef struct {
    const char *axis;
    const char *method;
    int buf;
    int fir;
    int sources;
    int swap;
    int depth;
    int k;
    double e2e_p50;
    double e2e_p99;
    double e2e_max;
    double ker_p50;
    double avx_p50;
    double deadline_us;
    long underruns;
    int ntimed;
    double snr;
    int valid;
} Row;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static double elapsed_us(const struct timespec *t0, const struct timespec *t1) {
    return (double)(t1->tv_sec - t0->tv_sec) * 1.0e6 + (double)(t1->tv_nsec - t0->tv_nsec) / 1.0e3;
}

static uint32_t source_seed(int source, int orient) {
    return CONV_SEED + (uint32_t)source * 0x10001u + (uint32_t)orient * 0x0100u;
}

static void build_inputs(int buf, int fir, int sources, int norient, int total_blocks, float **x_out,
                         float **h_orig_out, float **h_flip_out) {
    const int hist = fir - 1;
    const int nx = hist + total_blocks * buf;
    const size_t h_n = (size_t)norient * (size_t)sources * 2u * (size_t)fir;
    float *x = (float *)xmalloc((size_t)sources * (size_t)nx * sizeof(float));
    float *h_orig = (float *)xmalloc(h_n * sizeof(float));
    float *h_flip = (float *)xmalloc(h_n * sizeof(float));
    int orient, s, ch;
    float scratch_l[4096];
    float scratch_r[4096];

    if (fir > 4096) {
        fprintf(stderr, "fir %d exceeds the scratch HRIR buffer\n", fir);
        exit(1);
    }
    for (s = 0; s < sources; ++s) {
        signal_generate_at(x + (size_t)s * (size_t)nx, nx, hist, scratch_l, scratch_r, fir,
                           source_seed(s, 0));
    }
    for (orient = 0; orient < norient; ++orient) {
        for (s = 0; s < sources; ++s) {
            float *base = h_orig + (((size_t)orient * (size_t)sources + (size_t)s) * 2u) * (size_t)fir;
            hrir_generate(base, base + fir, fir, source_seed(s, orient));
            for (ch = 0; ch < 2; ++ch) {
                float *dst = h_flip + (((size_t)orient * (size_t)sources + (size_t)s) * 2u + (size_t)ch) *
                                         (size_t)fir;
                hrir_flip(base + (size_t)ch * (size_t)fir, dst, fir);
            }
        }
    }
    *x_out = x;
    *h_orig_out = h_orig;
    *h_flip_out = h_flip;
}

static double snr_against_ref(const float *x, const float *h_flip, const float *y, int buf, int fir,
                              int sources, int norient, int swap, int warmup, int measured) {
    const int hist = fir - 1;
    const int nx = hist + (warmup + measured) * buf;
    double signal = 0.0;
    double noise = 0.0;
    double *ext = (double *)xmalloc((size_t)(hist + buf) * sizeof(double));
    double *ref = (double *)xmalloc((size_t)buf * sizeof(double));
    double *h = (double *)xmalloc((size_t)fir * sizeof(double));
    int i, s, ch, j;

    for (i = 0; i < measured; ++i) {
        const int abs_index = warmup + i;
        const int orient = swap ? (abs_index % norient) : 0;
        for (s = 0; s < sources; ++s) {
            const float *xs = x + (size_t)s * (size_t)nx + (size_t)abs_index * (size_t)buf;
            const float *hf = h_flip + (((size_t)orient * (size_t)sources + (size_t)s) * 2u) * (size_t)fir;
            const float *ys = y + ((size_t)i * (size_t)sources + (size_t)s) * 2u * (size_t)buf;
            for (j = 0; j < hist + buf; ++j) {
                ext[j] = (double)xs[j];
            }
            for (ch = 0; ch < 2; ++ch) {
                int k;
                for (k = 0; k < fir; ++k) {
                    h[k] = (double)hf[(size_t)ch * (size_t)fir + (size_t)k];
                }
                conv_ref_f64(ext, h, ref, buf, fir);
                for (j = 0; j < buf; ++j) {
                    double test = (double)ys[(size_t)ch * (size_t)buf + (size_t)j];
                    double err = test - ref[j];
                    signal += ref[j] * ref[j];
                    noise += err * err;
                }
            }
        }
    }
    free(ext);
    free(ref);
    free(h);
    if (noise == 0.0) {
        return INFINITY;
    }
    if (signal == 0.0) {
        return -INFINITY;
    }
    return 10.0 * log10(signal / noise);
}

static void time_avx(const float *x, const float *h_flip, int buf, int fir, int sources, int norient,
                     int swap, int warmup, int measured, double *us) {
    const int hist = fir - 1;
    const int nx = hist + (warmup + measured) * buf;
    float *scratch = (float *)xmalloc((size_t)buf * sizeof(float));
    int i, s, ch;

    for (i = 0; i < warmup; ++i) {
        for (s = 0; s < sources; ++s) {
            const float *hf = h_flip + (size_t)s * 2u * (size_t)fir;
            const float *ext = x + (size_t)s * (size_t)nx + (size_t)i * (size_t)buf;
            for (ch = 0; ch < 2; ++ch) {
                conv_avx2(ext, hf + (size_t)ch * (size_t)fir, scratch, buf, fir);
            }
        }
    }
    for (i = 0; i < measured; ++i) {
        struct timespec t0, t1;
        const int abs_index = warmup + i;
        const int orient = swap ? (abs_index % norient) : 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (s = 0; s < sources; ++s) {
            const float *hf =
                h_flip + (((size_t)orient * (size_t)sources + (size_t)s) * 2u) * (size_t)fir;
            const float *ext = x + (size_t)s * (size_t)nx + (size_t)abs_index * (size_t)buf;
            for (ch = 0; ch < 2; ++ch) {
                conv_avx2(ext, hf + (size_t)ch * (size_t)fir, scratch, buf, fir);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        us[i] = elapsed_us(&t0, &t1);
    }
    free(scratch);
}

static long count_underruns(const double *us, int n, double deadline) {
    long count = 0;
    int i;
    for (i = 0; i < n; ++i) {
        if (us[i] > deadline) {
            ++count;
        }
    }
    return count;
}

static int self_check(void) {
    const int buf = 32;
    const int fir = 32;
    const int hist = fir - 1;
    const int warmup = 4;
    const int measured = 8;
    const int total = warmup + measured;
    const int nx = hist + total * buf;
    float h_l[32], h_r[32];
    float h_flip[64];
    float h_orig[64];
    float *x = (float *)xmalloc((size_t)nx * sizeof(float));
    float *y = (float *)xmalloc((size_t)measured * 2u * (size_t)buf * sizeof(float));
    double us_e[8], us_k[8];
    double snr;
    int groups = 0;

    signal_generate_at(x, nx, hist, h_l, h_r, fir, CONV_SEED);
    hrir_flip(h_l, h_flip, fir);
    hrir_flip(h_r, h_flip + fir, fir);
    memcpy(h_orig, h_l, sizeof(h_l));
    memcpy(h_orig + fir, h_r, sizeof(h_r));
    if (e2e_time_measure(buf, fir, 1, 1, 0, 1, x, h_flip, total, warmup, measured, y, us_e, us_k) != 0) {
        fprintf(stderr, "time self-check launch failed\n");
        return 1;
    }
    snr = snr_against_ref(x, h_flip, y, buf, fir, 1, 1, 0, warmup, measured);
    fprintf(stderr, "self-check time SNR %.2f dB\n", snr);
    if (!(snr > 90.0)) {
        fprintf(stderr, "time-domain end-to-end does not match the double reference\n");
        return 1;
    }
    if (e2e_fft_measure(buf, fir, 1, x, h_orig, total, warmup, measured, y, us_e, us_k, &groups) != 0) {
        fprintf(stderr, "fft self-check launch failed\n");
        return 1;
    }
    snr = snr_against_ref(x, h_flip, y, buf, fir, 1, 1, 0, warmup, measured);
    fprintf(stderr, "self-check fft SNR %.2f dB (%d groups)\n", snr, groups);
    if (!(snr > 70.0)) {
        fprintf(stderr, "cuFFT overlap-add does not match the double reference\n");
        return 1;
    }
    free(x);
    free(y);
    return 0;
}

static void run_case(Row *row) {
    const int warmup = CONV_WARMUP;
    const int measured = CONV_BLOCKS;
    const int total = warmup + measured;
    const int norient = row->swap ? kOrients : 1;
    const int ntimed = row->k > 1 ? measured / row->k : measured;
    float *x = NULL;
    float *h_orig = NULL;
    float *h_flip = NULL;
    float *y = NULL;
    double *us_e = NULL;
    double *us_k = NULL;
    double *us_avx = NULL;
    TimeStats ts_e, ts_k, ts_a;
    int groups = 0;
    int i;

    fprintf(stderr, "run %s %s buf=%d fir=%d src=%d swap=%d depth=%d k=%d\n", row->axis, row->method,
            row->buf, row->fir, row->sources, row->swap, row->depth, row->k);
    build_inputs(row->buf, row->fir, row->sources, norient, total, &x, &h_orig, &h_flip);
    y = (float *)xmalloc((size_t)measured * (size_t)row->sources * 2u * (size_t)row->buf * sizeof(float));
    us_e = (double *)xmalloc((size_t)ntimed * sizeof(double));
    us_k = (double *)xmalloc((size_t)ntimed * sizeof(double));
    us_avx = (double *)xmalloc((size_t)measured * sizeof(double));

    if (strcmp(row->method, "cufft") == 0) {
        if (e2e_fft_measure(row->buf, row->fir, row->k, x, h_orig, total, warmup, measured, y, us_e, us_k,
                            &groups) != 0) {
            fprintf(stderr, "fft measure failed\n");
            exit(1);
        }
    } else if (e2e_time_measure(row->buf, row->fir, row->sources, row->depth, row->swap, norient, x, h_flip,
                                total, warmup, measured, y, us_e, us_k) != 0) {
        fprintf(stderr, "time measure failed\n");
        exit(1);
    }

    row->deadline_us = (double)row->buf / (double)CONV_FS * 1.0e6;
    if (row->k > 1) {
        for (i = 0; i < ntimed; ++i) {
            us_e[i] += (double)(row->k - 1) * row->deadline_us;
        }
    }
    row->underruns = count_underruns(us_e, ntimed, row->deadline_us);
    row->ntimed = ntimed;
    time_percentiles(us_e, ntimed, &ts_e);
    time_percentiles(us_k, ntimed, &ts_k);
    row->e2e_p50 = ts_e.p50_us;
    row->e2e_p99 = ts_e.p99_us;
    row->e2e_max = ts_e.max_us;
    row->ker_p50 = ts_k.p50_us;

    time_avx(x, h_flip, row->buf, row->fir, row->sources, norient, row->swap, warmup, measured, us_avx);
    time_percentiles(us_avx, measured, &ts_a);
    row->avx_p50 = ts_a.p50_us;
    row->snr = snr_against_ref(x, h_flip, y, row->buf, row->fir, row->sources, norient, row->swap, warmup,
                               measured);
    row->valid = (row->e2e_p99 < row->deadline_us) && (row->e2e_p50 < row->avx_p50);

    free(x);
    free(h_orig);
    free(h_flip);
    free(y);
    free(us_e);
    free(us_k);
    free(us_avx);
}

static const Row *find_default(const Row *rows, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (strcmp(rows[i].axis, "buf") == 0 && rows[i].buf == 1024) {
            return &rows[i];
        }
    }
    return NULL;
}

static void print_boundary(const Row *rows, int n, const char *axis, const char *label) {
    int i;
    int any = 0;
    printf("- %s:", label);
    for (i = 0; i < n; ++i) {
        if (strcmp(rows[i].axis, axis) != 0) {
            continue;
        }
        if (rows[i].valid) {
            printf(" %d", strcmp(axis, "sources") == 0 ? rows[i].sources : (strcmp(axis, "buf") == 0 ? rows[i].buf : rows[i].fir));
            any = 1;
        }
    }
    if (!any) {
        printf(" 无");
    }
    printf("\n");
}

int main(void) {
    static const int k_buf[] = {128, 256, 512, 1024, 2048, 4096};
    static const int k_fir[] = {128, 256, 512, 1024, 2048, 4096};
    static const int k_src[] = {1, 2, 4, 8, 16, 32};
    Row rows[kMaxRows];
    int nrows = 0;
    int i;
    const Row *def;

    if (self_check() != 0) {
        return 1;
    }

    for (i = 0; i < 6; ++i) {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "buf";
        row.method = "time";
        row.buf = k_buf[i];
        row.fir = 512;
        row.sources = 1;
        row.depth = 1;
        row.k = 1;
        run_case(&row);
        rows[nrows++] = row;
    }
    for (i = 0; i < 6; ++i) {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "fir-time";
        row.method = "time";
        row.buf = 1024;
        row.fir = k_fir[i];
        row.sources = 1;
        row.depth = 1;
        row.k = 1;
        run_case(&row);
        rows[nrows++] = row;
    }
    for (i = 0; i < 6; ++i) {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "fir-fft";
        row.method = "cufft";
        row.buf = 1024;
        row.fir = k_fir[i];
        row.sources = 1;
        row.depth = 1;
        row.k = 1;
        run_case(&row);
        rows[nrows++] = row;
    }
    {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "fir-fft-k4";
        row.method = "cufft";
        row.buf = 1024;
        row.fir = 4096;
        row.sources = 1;
        row.depth = 1;
        row.k = 4;
        run_case(&row);
        rows[nrows++] = row;
    }
    for (i = 0; i < 6; ++i) {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "sources";
        row.method = "time";
        row.buf = 1024;
        row.fir = 512;
        row.sources = k_src[i];
        row.depth = 1;
        row.k = 1;
        run_case(&row);
        rows[nrows++] = row;
    }
    {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "azimuth";
        row.method = "time";
        row.buf = 1024;
        row.fir = 512;
        row.sources = 1;
        row.swap = 1;
        row.depth = 1;
        row.k = 1;
        run_case(&row);
        rows[nrows++] = row;
    }
    for (i = 2; i <= 3; ++i) {
        Row row;
        memset(&row, 0, sizeof(row));
        row.axis = "stream";
        row.method = "time";
        row.buf = 1024;
        row.fir = 512;
        row.sources = 1;
        row.depth = i;
        row.k = 1;
        run_case(&row);
        rows[nrows++] = row;
    }

    printf("# 第二阶段结果\n\n");
    printf("采样率 %d Hz。有效：端到端 p99 低于 `BUF/Fs`，且端到端 p50 快于同一配置的 AVX2。", CONV_FS);
    printf("K=4 的端到端时间含 `(K-1)*BUF/Fs` 的等待。历史留在显存，每块只搬运新样本和输出。\n\n");
    printf("| 轴 | 方法 | BUF | FIR | 声源 | 换系数 | 深度 | K | 端到端 p50 (us) | p99 (us) | max (us) | kernel p50 (us) | AVX2 p50 (us) | 播放时长 (us) | 欠载 | SNR (dB) | 有效 |\n");
    printf("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for (i = 0; i < nrows; ++i) {
        const Row *r = &rows[i];
        printf("| %s | %s | %d | %d | %d | %d | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %ld | %.2f | %s |\n",
               r->axis, r->method, r->buf, r->fir, r->sources, r->swap, r->depth, r->k, r->e2e_p50, r->e2e_p99,
               r->e2e_max, r->ker_p50, r->avx_p50, r->deadline_us, r->underruns, r->snr, r->valid ? "是" : "否");
    }

    def = find_default(rows, nrows);
    printf("\n## 结论\n\n");
    if (def) {
        printf("- 默认点 BUF=1024、FIR=512、1 声源、深度 1：端到端 p50 %.2f us，p99 %.2f us，AVX2 p50 %.2f us，播放时长 %.2f us，%s。kernel p50 %.2f us。\n",
               def->e2e_p50, def->e2e_p99, def->avx_p50, def->deadline_us, def->valid ? "有效" : "无效",
               def->ker_p50);
    }
    print_boundary(rows, nrows, "buf", "块长轴上有效的 BUF");
    print_boundary(rows, nrows, "fir-time", "阶数轴上时域有效的 FIR");
    print_boundary(rows, nrows, "sources", "声源轴上有效的声源数");
    printf("- cuFFT 相对时域 p50 更快的 FIR:");
    {
        int any = 0;
        for (i = 0; i < 6; ++i) {
            const Row *tm = NULL;
            const Row *ff = NULL;
            int j;
            for (j = 0; j < nrows; ++j) {
                if (strcmp(rows[j].axis, "fir-time") == 0 && rows[j].fir == k_fir[i]) {
                    tm = &rows[j];
                }
                if (strcmp(rows[j].axis, "fir-fft") == 0 && rows[j].fir == k_fir[i]) {
                    ff = &rows[j];
                }
            }
            if (tm && ff && ff->e2e_p50 < tm->e2e_p50) {
                printf(" %d", k_fir[i]);
                any = 1;
            }
        }
        if (!any) {
            printf(" 无");
        }
        printf("\n");
    }
    printf("- cuFFT 计算段（不含搬运）p50 低于时域 kernel 的 FIR:");
    {
        int any = 0;
        for (i = 0; i < 6; ++i) {
            const Row *tm = NULL;
            const Row *ff = NULL;
            int j;
            for (j = 0; j < nrows; ++j) {
                if (strcmp(rows[j].axis, "fir-time") == 0 && rows[j].fir == k_fir[i]) {
                    tm = &rows[j];
                }
                if (strcmp(rows[j].axis, "fir-fft") == 0 && rows[j].fir == k_fir[i]) {
                    ff = &rows[j];
                }
            }
            if (tm && ff && ff->ker_p50 < tm->ker_p50) {
                printf(" %d", k_fir[i]);
                any = 1;
            }
        }
        if (!any) {
            printf(" 无");
        }
        printf("。端到端仍未在扫描范围内超过时域。\n");
    }
    for (i = 0; i < nrows; ++i) {
        if (strcmp(rows[i].axis, "fir-fft-k4") == 0) {
            printf("- FIR=4096 攒 4 块再 FFT：端到端 p50 %.2f us（含 3 块等待），kernel/计算 p50 %.2f us，播放时长仍按 1 块 %.2f us，%s。\n",
                   rows[i].e2e_p50, rows[i].ker_p50, rows[i].deadline_us, rows[i].valid ? "有效" : "无效");
        }
        if (strcmp(rows[i].axis, "azimuth") == 0 && def) {
            printf("- 每块换 HRIR 的端到端 p50 %.2f us，比系数固定高 %.2f us。\n", rows[i].e2e_p50,
                   rows[i].e2e_p50 - def->e2e_p50);
        }
    }
    printf("- stream 深度:");
    if (def) {
        printf(" 1 -> %.2f us", def->e2e_p50);
    }
    for (i = 0; i < nrows; ++i) {
        if (strcmp(rows[i].axis, "stream") == 0) {
            printf("；%d -> %.2f us", rows[i].depth, rows[i].e2e_p50);
        }
    }
    printf("\n");
    printf("\n## 下一阶段输入\n\n");
    printf("- 实验三仍用 f32_unroll、块大小 512，在默认点 BUF=1024、FIR=512、1 声源上比较四条主机到设备通路。\n");
    printf("- 本阶段的端到端路径是历史留在显存、每块只搬运新样本和输出。通路对照不要改卷积内核。\n");
    printf("- `cudaHostRegister` 在第一阶段已经成功，注册路径可以测。\n");

    e2e_fft_shutdown();
    return 0;
}

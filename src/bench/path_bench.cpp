#define _POSIX_C_SOURCE 200809L

#include "conv.h"
#include "e2e.h"
#include "paths.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <time.h>

namespace {

struct Row {
    const char *name;
    int sources;
    int available;
    double e2e_p50;
    double e2e_p99;
    double e2e_max;
    double ker_p50;
    double copy_p50;
    double copy_frac;
    long bytes;
    long underruns;
    double snr;
    double avx_p50;
    int valid;
};

void *xmalloc(size_t bytes) {
    void *p = std::malloc(bytes);
    if (!p) {
        std::fprintf(stderr, "out of memory\n");
        std::exit(1);
    }
    return p;
}

double elapsed_us(const timespec &t0, const timespec &t1) {
    return static_cast<double>(t1.tv_sec - t0.tv_sec) * 1.0e6 +
           static_cast<double>(t1.tv_nsec - t0.tv_nsec) / 1.0e3;
}

uint32_t source_seed(int source) {
    return CONV_SEED + static_cast<uint32_t>(source) * 0x10001u;
}

void build_inputs(int buf, int fir, int sources, int total, float **x_out, float **h_flip_out) {
    const int hist = fir - 1;
    const int nx = hist + total * buf;
    float *x = static_cast<float *>(xmalloc(static_cast<size_t>(sources) * nx * sizeof(float)));
    float *h_flip = static_cast<float *>(xmalloc(static_cast<size_t>(sources) * 2u * fir * sizeof(float)));
    float h_l[4096];
    float h_r[4096];
    for (int s = 0; s < sources; ++s) {
        signal_generate_at(x + static_cast<size_t>(s) * nx, nx, hist, h_l, h_r, fir, source_seed(s));
        hrir_flip(h_l, h_flip + (static_cast<size_t>(s) * 2u) * fir, fir);
        hrir_flip(h_r, h_flip + (static_cast<size_t>(s) * 2u + 1u) * fir, fir);
    }
    *x_out = x;
    *h_flip_out = h_flip;
}

double snr_against_ref(const float *x, const float *h_flip, const float *y, int buf, int fir, int sources,
                       int warmup, int measured) {
    const int hist = fir - 1;
    const int nx = hist + (warmup + measured) * buf;
    double signal = 0.0;
    double noise = 0.0;
    double *ext = static_cast<double *>(xmalloc(static_cast<size_t>(hist + buf) * sizeof(double)));
    double *ref = static_cast<double *>(xmalloc(static_cast<size_t>(buf) * sizeof(double)));
    double *h = static_cast<double *>(xmalloc(static_cast<size_t>(fir) * sizeof(double)));
    for (int i = 0; i < measured; ++i) {
        for (int s = 0; s < sources; ++s) {
            const float *xs = x + static_cast<size_t>(s) * nx + static_cast<size_t>(warmup + i) * buf;
            const float *hf = h_flip + static_cast<size_t>(s) * 2u * fir;
            const float *ys = y + (static_cast<size_t>(i) * sources + s) * 2u * buf;
            for (int j = 0; j < hist + buf; ++j) {
                ext[j] = xs[j];
            }
            for (int ch = 0; ch < 2; ++ch) {
                for (int k = 0; k < fir; ++k) {
                    h[k] = hf[ch * fir + k];
                }
                conv_ref_f64(ext, h, ref, buf, fir);
                for (int j = 0; j < buf; ++j) {
                    const double err = static_cast<double>(ys[ch * buf + j]) - ref[j];
                    signal += ref[j] * ref[j];
                    noise += err * err;
                }
            }
        }
    }
    std::free(ext);
    std::free(ref);
    std::free(h);
    if (noise == 0.0) {
        return INFINITY;
    }
    return 10.0 * std::log10(signal / noise);
}

double time_avx(const float *x, const float *h_flip, int buf, int fir, int sources, int warmup, int measured) {
    const int hist = fir - 1;
    const int nx = hist + (warmup + measured) * buf;
    float *scratch = static_cast<float *>(xmalloc(static_cast<size_t>(buf) * sizeof(float)));
    double *us = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    for (int i = 0; i < measured; ++i) {
        timespec t0{};
        timespec t1{};
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < sources; ++s) {
            const float *ext = x + static_cast<size_t>(s) * nx + static_cast<size_t>(warmup + i) * buf;
            const float *hf = h_flip + static_cast<size_t>(s) * 2u * fir;
            conv_avx2(ext, hf, scratch, buf, fir);
            conv_avx2(ext, hf + fir, scratch, buf, fir);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        us[i] = elapsed_us(t0, t1);
    }
    TimeStats ts{};
    time_percentiles(us, measured, &ts);
    std::free(scratch);
    std::free(us);
    return ts.p50_us;
}

long count_underruns(const double *us, int n, double deadline) {
    long count = 0;
    for (int i = 0; i < n; ++i) {
        if (us[i] > deadline) {
            ++count;
        }
    }
    return count;
}

int count_lines(const char *path) {
    std::ifstream in(path);
    int lines = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++lines;
    }
    return lines;
}

void finish_row(Row *row, double *us_e2e, double *us_kernel, double *us_copy, const float *x, const float *h_flip,
                const float *y, int buf, int fir, int sources, int warmup, int measured, double avx_p50) {
    TimeStats e2e{};
    TimeStats ker{};
    TimeStats copy{};
    row->underruns = count_underruns(us_e2e, measured, static_cast<double>(buf) / CONV_FS * 1.0e6);
    time_percentiles(us_e2e, measured, &e2e);
    time_percentiles(us_kernel, measured, &ker);
    time_percentiles(us_copy, measured, &copy);
    row->e2e_p50 = e2e.p50_us;
    row->e2e_p99 = e2e.p99_us;
    row->e2e_max = e2e.max_us;
    row->ker_p50 = ker.p50_us;
    row->copy_p50 = copy.p50_us;
    row->copy_frac = e2e.p50_us > 0.0 ? copy.p50_us / e2e.p50_us : 0.0;
    row->snr = snr_against_ref(x, h_flip, y, buf, fir, sources, warmup, measured);
    row->avx_p50 = avx_p50;
    const double deadline = static_cast<double>(buf) / CONV_FS * 1.0e6;
    row->valid = row->available && row->e2e_p99 < deadline && row->e2e_p50 < avx_p50;
}

int self_check() {
    constexpr int buf = 32;
    constexpr int fir = 32;
    constexpr int hist = fir - 1;
    constexpr int warmup = 2;
    constexpr int measured = 4;
    constexpr int total = warmup + measured;
    const int nx = hist + total * buf;
    float h_l[32];
    float h_r[32];
    float h_flip[64];
    float *x = static_cast<float *>(xmalloc(static_cast<size_t>(nx) * sizeof(float)));
    float *y = static_cast<float *>(xmalloc(static_cast<size_t>(measured) * 2u * buf * sizeof(float)));
    double us_e[4];
    double us_k[4];
    double us_c[4];
    long bytes = 0;
    signal_generate_at(x, nx, hist, h_l, h_r, fir, CONV_SEED);
    hrir_flip(h_l, h_flip, fir);
    hrir_flip(h_r, h_flip + fir, fir);
    if (path_pageable_measure(buf, fir, 1, x, h_flip, total, warmup, measured, y, us_e, us_k, us_c, &bytes) !=
        0) {
        std::fprintf(stderr, "pageable self-check failed\n");
        return 1;
    }
    const double snr = snr_against_ref(x, h_flip, y, buf, fir, 1, warmup, measured);
    std::fprintf(stderr, "self-check pageable SNR %.2f dB\n", snr);
    std::free(x);
    std::free(y);
    return snr > 90.0 ? 0 : 1;
}

void run_window_path(Row *row, int kind, int buf, int fir, int sources, int warmup, int measured, double avx) {
    const int total = warmup + measured;
    float *x = nullptr;
    float *h_flip = nullptr;
    build_inputs(buf, fir, sources, total, &x, &h_flip);
    float *y = static_cast<float *>(
        xmalloc(static_cast<size_t>(measured) * sources * 2u * buf * sizeof(float)));
    double *us_e = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    double *us_k = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    double *us_c = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    long bytes = 0;
    int rc = 1;
    if (kind == 0) {
        rc = path_pageable_measure(buf, fir, sources, x, h_flip, total, warmup, measured, y, us_e, us_k, us_c,
                                   &bytes);
    } else if (kind == 1) {
        rc = path_async_measure(buf, fir, sources, x, h_flip, total, warmup, measured, y, us_e, us_k, us_c,
                                &bytes);
    } else {
        rc = path_mapped_measure(buf, fir, sources, x, h_flip, total, warmup, measured, y, us_e, us_k, us_c,
                                 &bytes);
    }
    row->sources = sources;
    row->bytes = bytes;
    row->available = rc == 0;
    if (rc == 0) {
        finish_row(row, us_e, us_k, us_c, x, h_flip, y, buf, fir, sources, warmup, measured, avx);
    }
    std::free(x);
    std::free(h_flip);
    std::free(y);
    std::free(us_e);
    std::free(us_k);
    std::free(us_c);
}

void run_resident(Row *row, int buf, int fir, int sources, int warmup, int measured, double avx) {
    const int total = warmup + measured;
    float *x = nullptr;
    float *h_flip = nullptr;
    build_inputs(buf, fir, sources, total, &x, &h_flip);
    float *y = static_cast<float *>(
        xmalloc(static_cast<size_t>(measured) * sources * 2u * buf * sizeof(float)));
    double *us_e = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    double *us_k = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    double *us_c = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    const int rc = e2e_time_measure(buf, fir, sources, 1, 0, 1, x, h_flip, total, warmup, measured, y, us_e, us_k);
    row->sources = sources;
    row->available = rc == 0;
    row->bytes = static_cast<long>(sources) * static_cast<long>(buf + 2 * buf) * static_cast<long>(sizeof(float));
    if (rc == 0) {
        for (int i = 0; i < measured; ++i) {
            us_c[i] = us_e[i] - us_k[i];
            if (us_c[i] < 0.0) {
                us_c[i] = 0.0;
            }
        }
        finish_row(row, us_e, us_k, us_c, x, h_flip, y, buf, fir, sources, warmup, measured, avx);
    }
    std::free(x);
    std::free(h_flip);
    std::free(y);
    std::free(us_e);
    std::free(us_k);
    std::free(us_c);
}

void run_opencl(Row *row, int buf, int fir, int sources, int warmup, int measured, double avx) {
    const int total = warmup + measured;
    float *x = nullptr;
    float *h_flip = nullptr;
    build_inputs(buf, fir, sources, total, &x, &h_flip);
    float *y = static_cast<float *>(
        xmalloc(static_cast<size_t>(measured) * sources * 2u * buf * sizeof(float)));
    double *us_e = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    double *us_k = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    double *us_c = static_cast<double *>(xmalloc(static_cast<size_t>(measured) * sizeof(double)));
    long bytes = 0;
    const int rc =
        opencl_resident_measure(buf, fir, sources, x, h_flip, total, warmup, measured, y, us_e, us_k, us_c, &bytes);
    row->sources = sources;
    row->bytes = bytes;
    row->available = rc == 0;
    if (rc == 0) {
        finish_row(row, us_e, us_k, us_c, x, h_flip, y, buf, fir, sources, warmup, measured, avx);
    }
    std::free(x);
    std::free(h_flip);
    std::free(y);
    std::free(us_e);
    std::free(us_k);
    std::free(us_c);
}

void print_row(const Row &row) {
    if (!row.available) {
        std::printf("| %s | %d | — | — | — | — | — | — | — | — | — | — | 环境限制 |\n", row.name, row.sources);
        return;
    }
    std::printf(
        "| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %ld | %ld | %.2f | %.2f | %s |\n", row.name,
        row.sources, row.e2e_p50, row.e2e_p99, row.e2e_max, row.ker_p50, row.copy_p50, row.copy_frac, row.bytes,
        row.underruns, row.snr, row.avx_p50, row.valid ? "是" : "否");
}

}  // namespace

int main() {
    path_enable_mapped_host();
    if (self_check() != 0) {
        std::fprintf(stderr, "window kernel self-check failed\n");
        return 1;
    }
    constexpr int buf = 1024;
    constexpr int fir = 512;
    constexpr int warmup = CONV_WARMUP;
    constexpr int measured = CONV_BLOCKS;
    float *x1 = nullptr;
    float *h1 = nullptr;
    build_inputs(buf, fir, 1, warmup + measured, &x1, &h1);
    const double avx1 = time_avx(x1, h1, buf, fir, 1, warmup, measured);
    std::free(x1);
    std::free(h1);
    float *x8 = nullptr;
    float *h8 = nullptr;
    build_inputs(buf, fir, 8, warmup + measured, &x8, &h8);
    const double avx8 = time_avx(x8, h8, buf, fir, 8, warmup, measured);
    std::free(x8);
    std::free(h8);

    Row rows[6]{};
    rows[0].name = "可分页同步拷贝";
    rows[1].name = "注册后异步拷贝";
    rows[2].name = "映射零拷贝";
    rows[3].name = "历史留在显存";
    rows[4].name = "可分页同步拷贝";
    rows[5].name = "历史留在显存";
    std::fprintf(stderr, "pageable x1\n");
    run_window_path(&rows[0], 0, buf, fir, 1, warmup, measured, avx1);
    std::fprintf(stderr, "async x1\n");
    run_window_path(&rows[1], 1, buf, fir, 1, warmup, measured, avx1);
    std::fprintf(stderr, "mapped x1\n");
    run_window_path(&rows[2], 2, buf, fir, 1, warmup, measured, avx1);
    std::fprintf(stderr, "resident x1\n");
    run_resident(&rows[3], buf, fir, 1, warmup, measured, avx1);
    std::fprintf(stderr, "pageable x8\n");
    run_window_path(&rows[4], 0, buf, fir, 8, warmup, measured, avx8);
    std::fprintf(stderr, "resident x8\n");
    run_resident(&rows[5], buf, fir, 8, warmup, measured, avx8);

    char ocl_msg[128] = {};
    const int ocl_ok = opencl_probe(ocl_msg, static_cast<int>(sizeof(ocl_msg))) == 0;
    Row ocl{};
    ocl.name = "OpenCL 历史留在显存";
    ocl.sources = 1;
    if (ocl_ok) {
        std::fprintf(stderr, "opencl\n");
        run_opencl(&ocl, buf, fir, 1, warmup, measured, avx1);
    }

    std::printf("# 第三阶段结果\n\n");
    std::printf("默认点 BUF=%d，FIR=%d，采样率 %d Hz，播放时长 %.2f us。内核为共享窗口上的展开卷积，块大小 512。",
                buf, fir, CONV_FS, static_cast<double>(buf) / CONV_FS * 1.0e6);
    std::printf("有效：p99 低于播放时长，且端到端 p50 快于 AVX2。\n\n");
    std::printf("| 通路 | 声源 | 端到端 p50 (us) | p99 (us) | max (us) | kernel p50 (us) | 拷贝 p50 (us) | 拷贝/端到端 | 字节/块 | 欠载 | SNR (dB) | AVX2 p50 (us) | 有效 |\n");
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for (int i = 0; i < 4; ++i) {
        print_row(rows[i]);
    }
    std::printf("\n## 8 声源补点\n\n");
    std::printf("| 通路 | 声源 | 端到端 p50 (us) | p99 (us) | max (us) | kernel p50 (us) | 拷贝 p50 (us) | 拷贝/端到端 | 字节/块 | 欠载 | SNR (dB) | AVX2 p50 (us) | 有效 |\n");
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    print_row(rows[4]);
    print_row(rows[5]);

    int best = -1;
    for (int i = 0; i < 4; ++i) {
        if (!rows[i].available) {
            continue;
        }
        if (best < 0 || rows[i].e2e_p50 < rows[best].e2e_p50) {
            best = i;
        }
    }
    std::printf("\n## 结论\n\n");
    if (best >= 0 && rows[0].available) {
        std::printf("- 默认点最短通路是 %s，端到端 p50 %.2f us，比可分页同步拷贝少 %.2f us。\n", rows[best].name,
                    rows[best].e2e_p50, rows[0].e2e_p50 - rows[best].e2e_p50);
    }
    std::printf("- 拷贝占端到端 p50 的比例:");
    for (int i = 0; i < 4; ++i) {
        if (rows[i].available) {
            std::printf(" %s %.2f", rows[i].name, rows[i].copy_frac);
        }
    }
    std::printf("\n");
    int any_valid = 0;
    std::printf("- 默认点快过 AVX2 的通路:");
    for (int i = 0; i < 4; ++i) {
        if (rows[i].valid) {
            std::printf(" %s", rows[i].name);
            any_valid = 1;
        }
    }
    if (!any_valid) {
        std::printf(" 无");
    }
    std::printf("\n");
    if (rows[4].available) {
        std::printf("- 8 声源可分页同步拷贝%s有效。历史留在显存%s有效。\n", rows[4].valid ? "" : "不",
                    rows[5].available && rows[5].valid ? "" : "不");
    }
    std::printf("- OpenCL：%s。\n", ocl_msg);
    if (ocl.available && rows[3].available) {
        const int ocl_lines = count_lines("src/opencl/fir_opencl.cpp");
        const int cuda_lines = count_lines("src/cuda/fir_e2e.cu");
        std::printf("- OpenCL 端到端 p99 %.2f us，CUDA 历史留在显存 p99 %.2f us。实现行数 OpenCL %d，CUDA `fir_e2e.cu` %d。\n",
                    ocl.e2e_p99, rows[3].e2e_p99, ocl_lines, cuda_lines);
    }
    return 0;
}

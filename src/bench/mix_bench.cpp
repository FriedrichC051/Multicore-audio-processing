#include "conv.h"
#include "hrir_field.hpp"
#include "motion.hpp"

#include <cstdio>
#include <vector>

namespace {

constexpr int kFir = 512;
constexpr int kBuf = 1024;
constexpr int kHist = kFir - 1;
constexpr int kWarmup = 20;
constexpr int kMeasured = 1000;
constexpr int kHot = 8;
constexpr int kStep = 5;
constexpr double kDeadlineUs = 1024.0 / 48000.0 * 1.0e6;

struct Row {
    int sources;
    int mix_outputs;
    double p50;
    double p99;
    double max_us;
    int underruns;
    double snr_numeric;
    double snr_wave;
};

int nx_for(int total) { return kHist + total * kBuf; }

int linearity_self_check() {
    const int fir = 4;
    const int buf = 4;
    double ext[7] = {0, 0, 0, 1, 0, 0, 0};
    double ha[4] = {4, 3, 2, 1};
    double hb[4] = {1, 0, 0, 2};
    double mixed[4];
    double y_coeff[4];
    double ya[4];
    double yb[4];
    const double w = 0.3;
    for (int i = 0; i < fir; ++i) {
        mixed[i] = (1.0 - w) * ha[i] + w * hb[i];
    }
    conv_ref_f64(ext, mixed, y_coeff, buf, fir);
    conv_ref_f64(ext, ha, ya, buf, fir);
    conv_ref_f64(ext, hb, yb, buf, fir);
    for (int j = 0; j < buf; ++j) {
        const double y_out = (1.0 - w) * ya[j] + w * yb[j];
        const double err = y_out - y_coeff[j];
        if (err > 1e-9 || err < -1e-9) {
            return 1;
        }
    }
    return 0;
}

double numeric_snr(const float* x, const float* y_out, int sources, int total, int mix_outputs) {
    int hot_grid[16];
    const int hot_n = select_hot_grid(kHot, kStep, hot_grid);
    std::vector<float> az(static_cast<size_t>(total) * sources);
    fill_azimuths(sources, total, kWarmup + 100, az.data());
    const int nx = nx_for(total);
    std::vector<double> ref_l;
    std::vector<double> ref_r;
    std::vector<double> test_l;
    std::vector<double> test_r;
    std::vector<double> ext(static_cast<size_t>(kHist + kBuf));
    std::vector<double> h_l(kFir);
    std::vector<double> h_r(kFir);
    std::vector<double> y_l(kBuf);
    std::vector<double> y_r(kBuf);
    std::vector<double> y_l_b(kBuf);
    std::vector<double> y_r_b(kBuf);
    std::vector<float> coeff(2u * kFir);
    std::vector<float> left_a(kFir);
    std::vector<float> right_a(kFir);
    std::vector<float> left_b(kFir);
    std::vector<float> right_b(kFir);
    std::vector<float> flip_l(kFir);
    std::vector<float> flip_r(kFir);
    std::vector<float> flip_lb(kFir);
    std::vector<float> flip_rb(kFir);
    for (int pick = 0; pick < kMeasured; pick += 20) {
        const int b = kWarmup + pick;
        for (int s = 0; s < sources; ++s) {
            const float angle = az[static_cast<size_t>(b) * sources + s];
            for (int i = 0; i < kHist + kBuf; ++i) {
                ext[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(s) * nx + static_cast<size_t>(b) * kBuf + i];
            }
            if (mix_outputs == 0) {
                reference_flipped(angle, hot_grid, hot_n, 0, kStep, coeff.data(), kFir);
                for (int i = 0; i < kFir; ++i) {
                    h_l[static_cast<size_t>(i)] = coeff[static_cast<size_t>(i)];
                    h_r[static_cast<size_t>(i)] = coeff[static_cast<size_t>(kFir + i)];
                }
                conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
                conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
            } else {
                float weight = 0.0f;
                int kind = 0;
                grid_endpoints(angle, hot_grid, hot_n, kStep, 0, left_a.data(), right_a.data(), left_b.data(),
                               right_b.data(), &weight, &kind, kFir);
                hrir_flip_buffer(left_a.data(), flip_l.data(), kFir);
                hrir_flip_buffer(right_a.data(), flip_r.data(), kFir);
                hrir_flip_buffer(left_b.data(), flip_lb.data(), kFir);
                hrir_flip_buffer(right_b.data(), flip_rb.data(), kFir);
                for (int i = 0; i < kFir; ++i) {
                    h_l[static_cast<size_t>(i)] = flip_l[static_cast<size_t>(i)];
                    h_r[static_cast<size_t>(i)] = flip_r[static_cast<size_t>(i)];
                }
                conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
                conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
                for (int i = 0; i < kFir; ++i) {
                    h_l[static_cast<size_t>(i)] = flip_lb[static_cast<size_t>(i)];
                    h_r[static_cast<size_t>(i)] = flip_rb[static_cast<size_t>(i)];
                }
                conv_ref_f64(ext.data(), h_l.data(), y_l_b.data(), kBuf, kFir);
                conv_ref_f64(ext.data(), h_r.data(), y_r_b.data(), kBuf, kFir);
                for (int j = 0; j < kBuf; ++j) {
                    y_l[static_cast<size_t>(j)] =
                        (1.0 - weight) * y_l[static_cast<size_t>(j)] + weight * y_l_b[static_cast<size_t>(j)];
                    y_r[static_cast<size_t>(j)] =
                        (1.0 - weight) * y_r[static_cast<size_t>(j)] + weight * y_r_b[static_cast<size_t>(j)];
                }
            }
            const float* block = y_out + (static_cast<size_t>(pick) * sources + s) * 2u * kBuf;
            for (int j = 0; j < kBuf; ++j) {
                ref_l.push_back(y_l[static_cast<size_t>(j)]);
                ref_r.push_back(y_r[static_cast<size_t>(j)]);
                test_l.push_back(block[j]);
                test_r.push_back(block[kBuf + j]);
            }
        }
    }
    return snr_db(ref_l.data(), test_l.data(), ref_r.data(), test_r.data(), static_cast<int>(ref_l.size()));
}

double wave_snr(const float* coeff_y, const float* mix_y, int sources) {
    const int n = kMeasured * sources * kBuf;
    std::vector<double> ref_l(static_cast<size_t>(n));
    std::vector<double> ref_r(static_cast<size_t>(n));
    std::vector<double> test_l(static_cast<size_t>(n));
    std::vector<double> test_r(static_cast<size_t>(n));
    for (int m = 0; m < kMeasured; ++m) {
        for (int s = 0; s < sources; ++s) {
            const float* a = coeff_y + (static_cast<size_t>(m) * sources + s) * 2u * kBuf;
            const float* b = mix_y + (static_cast<size_t>(m) * sources + s) * 2u * kBuf;
            for (int j = 0; j < kBuf; ++j) {
                const int i = (m * sources + s) * kBuf + j;
                ref_l[static_cast<size_t>(i)] = a[j];
                ref_r[static_cast<size_t>(i)] = a[kBuf + j];
                test_l[static_cast<size_t>(i)] = b[j];
                test_r[static_cast<size_t>(i)] = b[kBuf + j];
            }
        }
    }
    return snr_db(ref_l.data(), test_l.data(), ref_r.data(), test_r.data(), n);
}

int run_timed(const float* x, int sources, int mix_outputs, float* y, Row* row) {
    const int total = kWarmup + kMeasured;
    std::vector<double> us(static_cast<size_t>(kMeasured));
    long uploads = 0;
    fprintf(stderr, "mix sources=%d outputs=%d\n", sources, mix_outputs);
    if (motion_execute(x, sources, total, kWarmup, kMeasured, kHot, 1, kStep, mix_outputs, y, us.data(),
                       &uploads) != 0) {
        fprintf(stderr, "motion_execute failed\n");
        return 1;
    }
    row->sources = sources;
    row->mix_outputs = mix_outputs;
    row->underruns = 0;
    for (double sample : us) {
        if (sample > kDeadlineUs) {
            ++row->underruns;
        }
    }
    std::vector<double> sorted = us;
    TimeStats stats{};
    time_percentiles(sorted.data(), kMeasured, &stats);
    row->p50 = stats.p50_us;
    row->p99 = stats.p99_us;
    row->max_us = stats.max_us;
    row->snr_numeric = numeric_snr(x, y, sources, total, mix_outputs);
    row->snr_wave = 0.0;
    if (row->snr_numeric < 100.0) {
        fprintf(stderr, "numeric snr %.2f\n", row->snr_numeric);
        return 1;
    }
    return 0;
}

const char* mix_name(int mix_outputs) { return mix_outputs == 0 ? "先混合系数" : "先卷积再混合输出"; }

}  // namespace

int main() {
    if (hrir_field_self_check() != 0 || linearity_self_check() != 0) {
        fprintf(stderr, "self-check failed\n");
        return 1;
    }
    const int source_sets[] = {8, 32};
    std::vector<Row> rows;
    for (int sources : source_sets) {
        const int total = kWarmup + kMeasured;
        const int nx = nx_for(total);
        std::vector<float> x(static_cast<size_t>(sources) * nx);
        std::vector<float> y_coeff(static_cast<size_t>(kMeasured) * sources * 2u * kBuf);
        std::vector<float> y_mix(static_cast<size_t>(kMeasured) * sources * 2u * kBuf);
        float unused_l[kFir];
        float unused_r[kFir];
        for (int s = 0; s < sources; ++s) {
            signal_generate_at(x.data() + static_cast<size_t>(s) * nx, nx, kHist, unused_l, unused_r, kFir,
                               0x00C0FFEEu + static_cast<uint32_t>(s) * 17u);
        }
        Row coeff{};
        Row mixed{};
        if (run_timed(x.data(), sources, 0, y_coeff.data(), &coeff) != 0) {
            return 1;
        }
        if (run_timed(x.data(), sources, 1, y_mix.data(), &mixed) != 0) {
            return 1;
        }
        const double wave = wave_snr(y_coeff.data(), y_mix.data(), sources);
        coeff.snr_wave = wave;
        mixed.snr_wave = wave;
        rows.push_back(coeff);
        rows.push_back(mixed);
    }

    printf("# 两种混合\n\n");
    printf("网格 5 度，热方向 %d 个，8 路和 32 路，运动与主实验相同。块长 1024，预热 %d 段，计时 %d 段。\n\n",
           kHot, kWarmup, kMeasured);
    printf("| 路数 | 混合 | p50 μs | p99 μs | 最大 μs | 超时段 | 相对双精度 SNR dB | 两种输出互比 SNR dB |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (const Row& row : rows) {
        printf("| %d | %s | %.2f | %.2f | %.2f | %d | %.2f | %.2f |\n", row.sources, mix_name(row.mix_outputs),
               row.p50, row.p99, row.max_us, row.underruns, row.snr_numeric, row.snr_wave);
    }
    printf("\n");
    for (int sources : source_sets) {
        const Row* coeff = nullptr;
        const Row* mixed = nullptr;
        for (const Row& row : rows) {
            if (row.sources != sources) {
                continue;
            }
            if (row.mix_outputs == 0) {
                coeff = &row;
            } else {
                mixed = &row;
            }
        }
        if (coeff == nullptr || mixed == nullptr) {
            continue;
        }
        const double gap = mixed->p50 - coeff->p50;
        printf("%d 路先混合系数 p50 %.2f μs，先卷积再混合输出 p50 %.2f μs，相差 %.2f μs。", sources, coeff->p50,
               mixed->p50, gap);
        if (gap > -10.0 && gap < 10.0) {
            printf("这一档看不出多一次卷积。");
        } else if (gap > 0.0) {
            printf("多出来的时间是多一次卷积的代价，不是听感差别。");
        } else {
            printf("先卷积再混合输出这一次更短。");
        }
        printf("两种输出互比 %.2f dB。\n", coeff->snr_wave);
    }
    printf("卷积是线性的，双精度里两种混合重合。表里的互比是显卡单精度两次实现的波形差。\n");
    printf("以上微秒是 WSL2 上的读数。\n");
    return 0;
}

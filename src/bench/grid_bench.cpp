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
constexpr double kDeadlineUs = 1024.0 / 48000.0 * 1.0e6;

struct Row {
    int sources;
    int step;
    double p50;
    double p99;
    double max_us;
    int underruns;
    long reuploads;
    long cold_blocks;
    long hit_blocks;
    double snr_numeric;
    double snr_approx;
};

int nx_for(int total) { return kHist + total * kBuf; }

void push_channel(std::vector<double>* ref, std::vector<double>* test, const double* y_ref, const float* y_test) {
    for (int j = 0; j < kBuf; ++j) {
        ref->push_back(y_ref[j]);
        test->push_back(y_test[j]);
    }
}

void snr_pair(const float* x, const float* y_out, int sources, int total, int step, Row* row) {
    int hot_grid[16];
    const int hot_n = select_hot_grid(kHot, step, hot_grid);
    std::vector<float> az(static_cast<size_t>(total) * sources);
    fill_azimuths(sources, total, kWarmup + 100, az.data());
    const int nx = nx_for(total);
    std::vector<double> num_ref_l;
    std::vector<double> num_ref_r;
    std::vector<double> num_test_l;
    std::vector<double> num_test_r;
    std::vector<double> app_ref_l;
    std::vector<double> app_ref_r;
    std::vector<double> app_test_l;
    std::vector<double> app_test_r;
    std::vector<double> ext(static_cast<size_t>(kHist + kBuf));
    std::vector<double> h_l(kFir);
    std::vector<double> h_r(kFir);
    std::vector<double> y_l(kBuf);
    std::vector<double> y_r(kBuf);
    std::vector<float> coeff(2u * kFir);
    row->cold_blocks = 0;
    row->hit_blocks = 0;
    for (int b = kWarmup; b < total; ++b) {
        for (int s = 0; s < sources; ++s) {
            TierHit hit{};
            classify_azimuth(az[static_cast<size_t>(b) * sources + s], hot_grid, hot_n, step, &hit);
            if (hit.kind == kTierCold) {
                ++row->cold_blocks;
            } else {
                ++row->hit_blocks;
            }
        }
    }
    for (int pick = 0; pick < kMeasured; pick += 20) {
        const int b = kWarmup + pick;
        for (int s = 0; s < sources; ++s) {
            const float angle = az[static_cast<size_t>(b) * sources + s];
            TierHit hit{};
            classify_azimuth(angle, hot_grid, hot_n, step, &hit);
            for (int i = 0; i < kHist + kBuf; ++i) {
                ext[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(s) * nx + static_cast<size_t>(b) * kBuf + i];
            }
            reference_flipped(angle, hot_grid, hot_n, 0, step, coeff.data(), kFir);
            for (int i = 0; i < kFir; ++i) {
                h_l[static_cast<size_t>(i)] = coeff[static_cast<size_t>(i)];
                h_r[static_cast<size_t>(i)] = coeff[static_cast<size_t>(kFir + i)];
            }
            conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
            conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
            const float* block = y_out + (static_cast<size_t>(pick) * sources + s) * 2u * kBuf;
            push_channel(&num_ref_l, &num_test_l, y_l.data(), block);
            push_channel(&num_ref_r, &num_test_r, y_r.data(), block + kBuf);
            if (hit.kind == kTierCold) {
                continue;
            }
            reference_flipped(angle, hot_grid, hot_n, 1, step, coeff.data(), kFir);
            for (int i = 0; i < kFir; ++i) {
                h_l[static_cast<size_t>(i)] = coeff[static_cast<size_t>(i)];
                h_r[static_cast<size_t>(i)] = coeff[static_cast<size_t>(kFir + i)];
            }
            conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
            conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
            push_channel(&app_ref_l, &app_test_l, y_l.data(), block);
            push_channel(&app_ref_r, &app_test_r, y_r.data(), block + kBuf);
        }
    }
    row->snr_numeric =
        snr_db(num_ref_l.data(), num_test_l.data(), num_ref_r.data(), num_test_r.data(),
               static_cast<int>(num_ref_l.size()));
    row->snr_approx =
        snr_db(app_ref_l.data(), app_test_l.data(), app_ref_r.data(), app_test_r.data(),
               static_cast<int>(app_ref_l.size()));
}

int run_row(int sources, int step, Row* row) {
    const int total = kWarmup + kMeasured;
    const int nx = nx_for(total);
    std::vector<float> x(static_cast<size_t>(sources) * nx);
    std::vector<float> y(static_cast<size_t>(kMeasured) * sources * 2u * kBuf);
    std::vector<double> us(static_cast<size_t>(kMeasured));
    float unused_l[kFir];
    float unused_r[kFir];
    for (int s = 0; s < sources; ++s) {
        signal_generate_at(x.data() + static_cast<size_t>(s) * nx, nx, kHist, unused_l, unused_r, kFir,
                           0x00C0FFEEu + static_cast<uint32_t>(s) * 17u);
    }
    long uploads = 0;
    fprintf(stderr, "grid sources=%d step=%d\n", sources, step);
    if (motion_execute(x.data(), sources, total, kWarmup, kMeasured, kHot, 1, step, 0, y.data(), us.data(),
                       &uploads) != 0) {
        fprintf(stderr, "motion_execute failed\n");
        return 1;
    }
    row->sources = sources;
    row->step = step;
    row->reuploads = uploads;
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
    snr_pair(x.data(), y.data(), sources, total, step, row);
    if (row->snr_numeric < 100.0) {
        fprintf(stderr, "numeric snr %.2f\n", row->snr_numeric);
        return 1;
    }
    return 0;
}

const Row* find_row(const std::vector<Row>& rows, int sources, int step) {
    for (const Row& row : rows) {
        if (row.sources == sources && row.step == step) {
            return &row;
        }
    }
    return nullptr;
}

}  // namespace

int main() {
    if (hrir_field_self_check() != 0) {
        fprintf(stderr, "hrir field self-check failed\n");
        return 1;
    }
    const int steps[] = {5, 15, 30};
    const int source_sets[] = {8, 32};
    std::vector<Row> rows;
    for (int sources : source_sets) {
        for (int step : steps) {
            Row row{};
            if (run_row(sources, step, &row) != 0) {
                return 1;
            }
            rows.push_back(row);
        }
    }

    int hot30[8];
    const int n30 = select_hot_grid(kHot, 30, hot30);
    printf("# 网格疏密\n\n");
    printf("热方向固定 %d 个，分层混合系数后再卷积。块长 1024，预热 %d 段，计时 %d 段。运动与主实验相同。\n",
           kHot, kWarmup, kMeasured);
    printf("30 度网格在 160 度以内只有 6 个点，8 个热方向里多出来的是");
    for (int i = 0; i < n30; ++i) {
        if (hot30[i] * 30 > 160) {
            printf(" %d 度", hot30[i] * 30);
        }
    }
    printf("，慢转不到那里。\n\n");
    printf("| 路数 | 步长 | p50 μs | p99 μs | 最大 μs | 超时段 | 未命中重送 | 未命中段 | 命中段 | 数值 SNR dB | 近似 SNR dB |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (const Row& row : rows) {
        printf("| %d | %d | %.2f | %.2f | %.2f | %d | %ld | %ld | %ld | %.2f | %.2f |\n", row.sources, row.step,
               row.p50, row.p99, row.max_us, row.underruns, row.reuploads, row.cold_blocks, row.hit_blocks,
               row.snr_numeric, row.snr_approx);
    }
    printf("\n");
    for (int sources : source_sets) {
        const Row* fine = find_row(rows, sources, 5);
        const Row* mid = find_row(rows, sources, 15);
        const Row* coarse = find_row(rows, sources, 30);
        if (fine == nullptr || mid == nullptr || coarse == nullptr) {
            continue;
        }
        printf("%d 路：5 度未命中重送 %ld 次、未命中段 %ld、近似信噪比 %.2f dB；", sources, fine->reuploads,
               fine->cold_blocks, fine->snr_approx);
        printf("15 度是 %ld 次、%ld 段、%.2f dB；", mid->reuploads, mid->cold_blocks, mid->snr_approx);
        printf("30 度是 %ld 次、%ld 段、%.2f dB。\n", coarse->reuploads, coarse->cold_blocks, coarse->snr_approx);
    }
    printf("近似信噪比只统计热区和温区命中的段，相对该方位的真实系数。冷区重送的是真实系数，没有算进这一列。\n");
    printf("数值信噪比相对同一次混合，用来确认实现。5 度的近似信噪比只有二十多 dB，是因为两套系数按比例相加抹不平延迟的跳变。\n");
    printf("从 5 度到 15 度，慢转仍被网格上的两个点夹住，未命中重送没有增加，近似信噪比下降。到 30 度，160 度附近被 150 度和冷区里的 180 度夹住，未命中重送和未命中段都变多，近似信噪比再降一截。\n");
    printf("以上微秒是 WSL2 上的读数。\n");
    return 0;
}

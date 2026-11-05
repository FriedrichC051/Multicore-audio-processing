#include "conv.h"
#include "hrir_field.hpp"
#include "motion.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kFir = 512;
constexpr int kBuf = 1024;
constexpr int kHist = kFir - 1;
constexpr int kWarmup = 20;
constexpr int kMeasured = 1000;
constexpr double kDeadlineUs = 1024.0 / 48000.0 * 1.0e6;

struct Row {
    int sources;
    int mode;
    int hot;
    double p50;
    double p99;
    double max_us;
    int underruns;
    long reuploads;
    long hot_blocks;
    long warm_blocks;
    long cold_blocks;
    double snr;
};

int nx_for(int total) { return kHist + total * kBuf; }

void count_tiers(int sources, int total, int warmup, int measured, int hot_count, long* hot_blocks,
                 long* warm_blocks, long* cold_blocks) {
    int hot_grid[16];
    const int hot_n = select_hot_grid(hot_count, 5, hot_grid);
    std::vector<float> az(static_cast<size_t>(total) * sources);
    fill_azimuths(sources, total, warmup + 100, az.data());
    *hot_blocks = 0;
    *warm_blocks = 0;
    *cold_blocks = 0;
    for (int b = warmup; b < warmup + measured; ++b) {
        for (int s = 0; s < sources; ++s) {
            TierHit hit{};
            classify_azimuth(az[static_cast<size_t>(b) * sources + s], hot_grid, hot_n, 5, &hit);
            if (hit.kind == kTierHot) {
                ++(*hot_blocks);
            } else if (hit.kind == kTierWarm) {
                ++(*warm_blocks);
            } else {
                ++(*cold_blocks);
            }
        }
    }
}

double snr_against_reference(const float* x, const float* y_out, int sources, int total, int warmup,
                             int measured, int hot_count, int mode) {
    int hot_grid[16];
    const int hot_n = select_hot_grid(hot_count, 5, hot_grid);
    std::vector<float> az(static_cast<size_t>(total) * sources);
    fill_azimuths(sources, total, warmup + 100, az.data());
    const int nx = nx_for(total);
    const int picks[] = {0, 50, 100, 250, 500, 999};
    std::vector<double> ref_l;
    std::vector<double> ref_r;
    std::vector<double> test_l;
    std::vector<double> test_r;
    std::vector<double> ext(static_cast<size_t>(kHist + kBuf));
    std::vector<double> h_l(kFir);
    std::vector<double> h_r(kFir);
    std::vector<double> y_l(kBuf);
    std::vector<double> y_r(kBuf);
    std::vector<float> coeff(2u * kFir);
    for (int pick : picks) {
        if (pick >= measured) {
            continue;
        }
        const int b = warmup + pick;
        for (int s = 0; s < sources; ++s) {
            reference_flipped(az[static_cast<size_t>(b) * sources + s], hot_grid, hot_n, mode == 0 ? 1 : 0, 5,
                              coeff.data(), kFir);
            for (int i = 0; i < kHist + kBuf; ++i) {
                ext[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(s) * nx + static_cast<size_t>(b) * kBuf + i];
            }
            for (int i = 0; i < kFir; ++i) {
                h_l[static_cast<size_t>(i)] = coeff[static_cast<size_t>(i)];
                h_r[static_cast<size_t>(i)] = coeff[static_cast<size_t>(kFir + i)];
            }
            conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
            conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
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

int run_row(int sources, int mode, int hot, int warmup, int measured, Row* row) {
    const int total = warmup + measured;
    const int nx = nx_for(total);
    std::vector<float> x(static_cast<size_t>(sources) * nx);
    std::vector<float> y(static_cast<size_t>(measured) * sources * 2u * kBuf);
    std::vector<double> us(static_cast<size_t>(measured));
    float unused_l[kFir];
    float unused_r[kFir];
    for (int s = 0; s < sources; ++s) {
        signal_generate_at(x.data() + static_cast<size_t>(s) * nx, nx, kHist, unused_l, unused_r, kFir,
                           0x00C0FFEEu + static_cast<uint32_t>(s) * 17u);
    }
    long uploads = 0;
    fprintf(stderr, "sources=%d mode=%d hot=%d blocks=%d\n", sources, mode, hot, measured);
    if (motion_execute(x.data(), sources, total, warmup, measured, hot, mode, 5, 0, y.data(), us.data(),
                       &uploads) != 0) {
        fprintf(stderr, "motion_execute failed\n");
        return 1;
    }
    row->sources = sources;
    row->mode = mode;
    row->hot = hot;
    row->reuploads = uploads;
    count_tiers(sources, total, warmup, measured, hot, &row->hot_blocks, &row->warm_blocks,
                &row->cold_blocks);
    row->underruns = 0;
    for (double sample : us) {
        if (sample > kDeadlineUs) {
            ++row->underruns;
        }
    }
    std::vector<double> sorted = us;
    TimeStats stats{};
    time_percentiles(sorted.data(), measured, &stats);
    row->p50 = stats.p50_us;
    row->p99 = stats.p99_us;
    row->max_us = stats.max_us;
    row->snr = snr_against_reference(x.data(), y.data(), sources, total, warmup, measured, hot, mode);
    return 0;
}

const char* mode_name(int mode) { return mode == 0 ? "reupload" : "tiered"; }

}  // namespace

int main() {
    if (hrir_field_self_check() != 0) {
        fprintf(stderr, "hrir field self-check failed\n");
        return 1;
    }
    const int limit = motion_hot_limit();
    if (limit < 2) {
        fprintf(stderr, "constant memory holds %d directions\n", limit);
        return 1;
    }
    Row gate{};
    if (run_row(1, 1, 4, 2, 8, &gate) != 0 || gate.snr < 100.0) {
        fprintf(stderr, "short tiered check snr %.2f\n", gate.snr);
        return 1;
    }
    if (run_row(1, 0, 4, 2, 8, &gate) != 0 || gate.snr < 100.0) {
        fprintf(stderr, "short reupload check snr %.2f\n", gate.snr);
        return 1;
    }

    const int requested[] = {2, 4, 8, 16};
    const int source_sets[] = {8, 32};
    std::vector<Row> rows;
    for (int sources : source_sets) {
        for (int want : requested) {
            const int hot = want > limit ? limit : want;
            for (int mode = 0; mode <= 1; ++mode) {
                Row row{};
                if (run_row(sources, mode, hot, kWarmup, kMeasured, &row) != 0) {
                    return 1;
                }
                rows.push_back(row);
            }
        }
    }
    Row single_up{};
    Row single_tier{};
    const int single_hot = 8 > limit ? limit : 8;
    if (run_row(1, 0, single_hot, kWarmup, kMeasured, &single_up) != 0) {
        return 1;
    }
    if (run_row(1, 1, single_hot, kWarmup, kMeasured, &single_tier) != 0) {
        return 1;
    }

    printf("# 冷热分层驻留\n\n");
    printf("块长 1024，采样率 48000 Hz，播放期限 %.2f μs。预热 %d 段，计时 %d 段。\n", kDeadlineUs, kWarmup,
           kMeasured);
    printf("水平网格 5 度。方位 [170, 190) 不在显存里，算冷区。热方向是从 0 度起连续的网格点。\n");
    printf("常量内存按 16 个方向、每个方向左右耳 512 点计为 64 KB。本次实际能放进 %d 个方向。\n\n", limit);
    printf("| 路数 | 策略 | 热方向 | p50 μs | p99 μs | 最大 μs | 超时段 | 主机重送 | 热段 | 温段 | 冷段 | SNR dB |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (const Row& row : rows) {
        printf("| %d | %s | %d | %.2f | %.2f | %.2f | %d | %ld | %ld | %ld | %ld | %.2f |\n", row.sources,
               mode_name(row.mode), row.hot, row.p50, row.p99, row.max_us, row.underruns, row.reuploads,
               row.hot_blocks, row.warm_blocks, row.cold_blocks, row.snr);
    }
    printf("\n单路、热方向 %d：重送 p50 %.2f μs，分层 p50 %.2f μs，差值 %.2f μs，占 21.33 ms 的 %.4f%%。\n",
           single_hot, single_up.p50, single_tier.p50, single_up.p50 - single_tier.p50,
           (single_up.p50 - single_tier.p50) / kDeadlineUs * 100.0);
    printf("单路信噪比：重送 %.2f dB，分层 %.2f dB。\n\n", single_up.snr, single_tier.snr);

    for (int sources : source_sets) {
        const Row* fastest = nullptr;
        double lo = 0.0;
        double hi = 0.0;
        long uploads = -1;
        bool uploads_same = true;
        for (const Row& row : rows) {
            if (row.sources != sources || row.mode != 1) {
                continue;
            }
            if (fastest == nullptr || row.p50 < fastest->p50) {
                fastest = &row;
            }
            if (uploads < 0) {
                uploads = row.reuploads;
                lo = hi = row.p50;
            } else {
                if (row.reuploads != uploads) {
                    uploads_same = false;
                }
                if (row.p50 < lo) {
                    lo = row.p50;
                }
                if (row.p50 > hi) {
                    hi = row.p50;
                }
            }
        }
        if (fastest == nullptr) {
            continue;
        }
        printf("%d 路分层的主机重送", sources);
        if (uploads_same) {
            printf("都是 %ld 次", uploads);
        } else {
            printf("随热方向个数变化");
        }
        printf("，p50 从 %.2f μs 到 %.2f μs。", lo, hi);
        if (hi - lo <= 10.0) {
            printf("各档只差 %.2f μs，热方向个数没有把端到端时间分开。\n", hi - lo);
        } else {
            printf("这一次较短的是 %d 个热方向，p50 %.2f μs。\n", fastest->hot, fastest->p50);
        }
    }
    printf("16 个热方向放进了常量内存。32 路里取端到端较短、重送不再下降的那一档，作为时延和常量内存占用的近似折中。\n");
    printf("分层的主机重送是甩进冷区的那两路，不是热方向个数变少了就少传。\n");
    printf("甩到 175 度的声源落在冷区，整套系数从主机送入，不用旁边网格上的系数硬混。\n");
    printf("单路省下的时间相对 21.33 ms 可以忽略。以上微秒是 WSL2 上的读数，不能当成裸机上的固定差距。\n");
    return 0;
}

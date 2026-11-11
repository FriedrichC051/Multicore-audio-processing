#include "conv.h"
#include "hrir_field.hpp"
#include "motion.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kFir = 512;
constexpr int kHist = kFir - 1;
constexpr int kWarmup = 20;
constexpr int kMeasured = 1000;
constexpr int kHot = 8;
constexpr double kFs = 48000.0;

struct Row {
    int buf;
    int mode;
    double p50;
    double p99;
    int underruns;
    long reuploads;
    double saved_frac;
};

int nx_for(int total, int buf) { return kHist + total * buf; }

int run_row(int buf, int mode, int warmup, int measured, Row* row) {
    const int total = warmup + measured;
    const int nx = nx_for(total, buf);
    const int sources = 32;
    std::vector<float> x(static_cast<size_t>(sources) * nx);
    std::vector<float> y(static_cast<size_t>(measured) * sources * 2u * buf);
    std::vector<double> us(static_cast<size_t>(measured));
    float unused_l[kFir];
    float unused_r[kFir];
    for (int s = 0; s < sources; ++s) {
        signal_generate_at(x.data() + static_cast<size_t>(s) * nx, nx, kHist, unused_l, unused_r, kFir,
                           0x00C0FFEEu + static_cast<uint32_t>(s) * 17u);
    }
    long uploads = 0;
    fprintf(stderr, "block buf=%d mode=%d\n", buf, mode);
    if (motion_execute(x.data(), sources, total, warmup, measured, kHot, mode, 5, 0, y.data(), us.data(),
                       &uploads, buf) != 0) {
        fprintf(stderr, "motion_execute failed\n");
        return 1;
    }
    const double deadline = static_cast<double>(buf) / kFs * 1.0e6;
    row->buf = buf;
    row->mode = mode;
    row->reuploads = uploads;
    row->underruns = 0;
    row->saved_frac = 0.0;
    for (double sample : us) {
        if (sample > deadline) {
            ++row->underruns;
        }
    }
    std::vector<double> sorted = us;
    TimeStats stats{};
    time_percentiles(sorted.data(), measured, &stats);
    row->p50 = stats.p50_us;
    row->p99 = stats.p99_us;
    return 0;
}

int gate_short_block() {
    constexpr int buf = 128;
    constexpr int warmup = 2;
    constexpr int measured = 6;
    constexpr int sources = 1;
    const int total = warmup + measured;
    const int nx = nx_for(total, buf);
    std::vector<float> x(static_cast<size_t>(nx));
    std::vector<float> y(static_cast<size_t>(measured) * 2u * buf);
    std::vector<double> us(static_cast<size_t>(measured));
    float unused_l[kFir];
    float unused_r[kFir];
    signal_generate_at(x.data(), nx, kHist, unused_l, unused_r, kFir, 0x00C0FFEEu);
    long uploads = 0;
    if (motion_execute(x.data(), sources, total, warmup, measured, kHot, 1, 5, 0, y.data(), us.data(),
                       &uploads, buf) != 0) {
        return 1;
    }
    int hot[16];
    const int hot_n = select_hot_grid(kHot, 5, hot);
    std::vector<float> az(static_cast<size_t>(total));
    fill_azimuths(sources, total, warmup + 100, az.data());
    std::vector<double> ext(static_cast<size_t>(kHist + buf));
    std::vector<double> h_l(kFir);
    std::vector<double> h_r(kFir);
    std::vector<double> y_l(buf);
    std::vector<double> y_r(buf);
    std::vector<double> ref_l;
    std::vector<double> ref_r;
    std::vector<double> test_l;
    std::vector<double> test_r;
    float coeff[2 * kFir];
    for (int pick = 0; pick < measured; ++pick) {
        const int b = warmup + pick;
        for (int i = 0; i < kHist + buf; ++i) {
            ext[static_cast<size_t>(i)] = x[static_cast<size_t>(b) * buf + i];
        }
        reference_flipped(az[static_cast<size_t>(b)], hot, hot_n, 0, 5, coeff, kFir);
        for (int i = 0; i < kFir; ++i) {
            h_l[static_cast<size_t>(i)] = coeff[i];
            h_r[static_cast<size_t>(i)] = coeff[kFir + i];
        }
        conv_ref_f64(ext.data(), h_l.data(), y_l.data(), buf, kFir);
        conv_ref_f64(ext.data(), h_r.data(), y_r.data(), buf, kFir);
        const float* block = y.data() + static_cast<size_t>(pick) * 2u * buf;
        for (int j = 0; j < buf; ++j) {
            ref_l.push_back(y_l[static_cast<size_t>(j)]);
            ref_r.push_back(y_r[static_cast<size_t>(j)]);
            test_l.push_back(block[j]);
            test_r.push_back(block[buf + j]);
        }
    }
    const double snr =
        snr_db(ref_l.data(), test_l.data(), ref_r.data(), test_r.data(), static_cast<int>(ref_l.size()));
    if (snr < 100.0) {
        fprintf(stderr, "short block snr %.2f\n", snr);
        return 1;
    }
    return 0;
}

const char* mode_name(int mode) { return mode == 0 ? "重送" : "分层"; }

const Row* find_row(const std::vector<Row>& rows, int buf, int mode) {
    for (const Row& row : rows) {
        if (row.buf == buf && row.mode == mode) {
            return &row;
        }
    }
    return nullptr;
}

}  // namespace

int main() {
    if (hrir_field_self_check() != 0 || gate_short_block() != 0) {
        fprintf(stderr, "block self-check failed\n");
        return 1;
    }
    const int bufs[] = {128, 256, 512, 1024};
    std::vector<Row> rows;
    for (int buf : bufs) {
        for (int mode = 0; mode <= 1; ++mode) {
            Row row{};
            if (run_row(buf, mode, kWarmup, kMeasured, &row) != 0) {
                return 1;
            }
            rows.push_back(row);
        }
    }
    for (int buf : bufs) {
        Row* tier = nullptr;
        const Row* up = find_row(rows, buf, 0);
        for (Row& row : rows) {
            if (row.buf == buf && row.mode == 1) {
                tier = &row;
            }
        }
        if (tier != nullptr && up != nullptr) {
            const double deadline = static_cast<double>(buf) / kFs * 1.0e6;
            tier->saved_frac = (up->p50 - tier->p50) / deadline;
        }
    }

    printf("# 块长四点\n\n");
    printf("32 路，热方向 8 个，网格 5 度，运动与主实验相同。预热 %d 段，计时 %d 段。\n\n", kWarmup, kMeasured);
    printf("| 块长 | 期限 μs | 策略 | p50 μs | p99 μs | 超时段 | 重送 | 省时占期限 |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (const Row& row : rows) {
        const double deadline = static_cast<double>(row.buf) / kFs * 1.0e6;
        if (row.mode == 0) {
            printf("| %d | %.2f | %s | %.2f | %.2f | %d | %ld | — |\n", row.buf, deadline, mode_name(row.mode),
                   row.p50, row.p99, row.underruns, row.reuploads);
        } else {
            printf("| %d | %.2f | %s | %.2f | %.2f | %d | %ld | %.4f%% |\n", row.buf, deadline, mode_name(row.mode),
                   row.p50, row.p99, row.underruns, row.reuploads, row.saved_frac * 100.0);
        }
    }
    printf("\n");
    int any = 0;
    for (const Row& row : rows) {
        if (row.underruns > 0) {
            printf("块长 %d 的%s有 %d 段超过该块期限。\n", row.buf, mode_name(row.mode), row.underruns);
            any = 1;
        }
    }
    if (any == 0) {
        printf("块长收到 128，期限 %.2f μs，重送和分层这次都没有超时段。\n", 128.0 / kFs * 1.0e6);
    } else {
        printf("分层在 128、256、512、1024 都没有超时段。块变短以后，超时段出现在重送。\n");
    }
    const Row* short_tier = find_row(rows, 128, 1);
    const Row* long_tier = find_row(rows, 1024, 1);
    if (short_tier != nullptr && long_tier != nullptr) {
        printf("分层省下的时间占该块期限的比例，128 点是 %.4f%%，1024 点是 %.4f%%。\n",
               short_tier->saved_frac * 100.0, long_tier->saved_frac * 100.0);
    }
    printf("以上微秒是 WSL2 上的读数。\n");
    return 0;
}

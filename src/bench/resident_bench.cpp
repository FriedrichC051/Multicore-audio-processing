#include "conv.h"
#include "hrir_field.hpp"
#include "motion.hpp"

#include <cmath>
#include <cstdio>
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
    int kind;
    int hot;
    double p50;
    double p99;
    int underruns;
    long reuploads;
    double snr_all;
    double snr_blend;
    int blend_samples;
    double snr_intended;
};

void mix_pair(const float* a_l, const float* a_r, const float* b_l, const float* b_r, float w, float* left,
              float* right) {
    for (int i = 0; i < kFir; ++i) {
        left[i] = (1.0f - w) * a_l[i] + w * b_l[i];
        right[i] = (1.0f - w) * a_r[i] + w * b_r[i];
    }
}

void intended_hrir(float az, const int* hot, int hot_n, int mode, float* left, float* right, int* blended) {
    *blended = 0;
    if (mode == 2) {
        TierHit bracket{};
        if (resident_bracket(az, hot, hot_n, 5, &bracket) == 1) {
            float a_l[kFir];
            float a_r[kFir];
            float b_l[kFir];
            float b_r[kFir];
            hrir_causal(static_cast<float>(bracket.index_a * 5), a_l, a_r, kFir);
            hrir_causal(static_cast<float>(bracket.index_b * 5), b_l, b_r, kFir);
            mix_pair(a_l, a_r, b_l, b_r, bracket.weight_b, left, right);
            *blended = 1;
            return;
        }
    } else if (mode == 1) {
        TierHit hit{};
        classify_azimuth(az, hot, hot_n, 5, &hit);
        if (hit.kind != kTierCold) {
            hrir_blend_linear(az, hot, hot_n, 5, left, right, kFir);
            *blended = 1;
            return;
        }
    }
    hrir_causal(az, left, right, kFir);
}

void accumulate(const double* test_l, const double* test_r, const double* ref_l, const double* ref_r, int n,
                double* signal, double* noise) {
    for (int j = 0; j < n; ++j) {
        const double dl = test_l[j] - ref_l[j];
        const double dr = test_r[j] - ref_r[j];
        *signal += ref_l[j] * ref_l[j] + ref_r[j] * ref_r[j];
        *noise += dl * dl + dr * dr;
    }
}

double finish_snr(double signal, double noise) {
    if (noise == 0.0) {
        return 200.0;
    }
    if (signal == 0.0) {
        return -200.0;
    }
    return 10.0 * std::log10(signal / noise);
}

void fill_snr(const float* x, const float* y_out, int sources, int total, int mode, int hot_count, Row* row) {
    int hot[16];
    const int hot_n = select_hot_grid(hot_count, 5, hot);
    std::vector<float> az(static_cast<size_t>(total) * sources);
    fill_azimuths(sources, total, kWarmup + 100, az.data());
    const int nx = kHist + total * kBuf;
    double all_sig = 0.0;
    double all_noise = 0.0;
    double blend_sig = 0.0;
    double blend_noise = 0.0;
    double intend_sig = 0.0;
    double intend_noise = 0.0;
    int blend_n = 0;
    std::vector<double> ext(static_cast<size_t>(kHist + kBuf));
    std::vector<double> h_l(kFir);
    std::vector<double> h_r(kFir);
    std::vector<double> y_l(kBuf);
    std::vector<double> y_r(kBuf);
    float coeff_l[kFir];
    float coeff_r[kFir];
    float true_l[kFir];
    float true_r[kFir];
    float flip_l[kFir];
    float flip_r[kFir];
    for (int pick = 0; pick < kMeasured; pick += 20) {
        const int b = kWarmup + pick;
        for (int s = 0; s < sources; ++s) {
            const float angle = az[static_cast<size_t>(b) * sources + s];
            for (int i = 0; i < kHist + kBuf; ++i) {
                ext[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(s) * nx + static_cast<size_t>(b) * kBuf + i];
            }
            int blended = 0;
            intended_hrir(angle, hot, hot_n, mode, coeff_l, coeff_r, &blended);
            hrir_causal(angle, true_l, true_r, kFir);
            hrir_flip_buffer(coeff_l, flip_l, kFir);
            hrir_flip_buffer(coeff_r, flip_r, kFir);
            for (int i = 0; i < kFir; ++i) {
                h_l[static_cast<size_t>(i)] = flip_l[i];
                h_r[static_cast<size_t>(i)] = flip_r[i];
            }
            conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
            conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
            const float* block = y_out + (static_cast<size_t>(pick) * sources + s) * 2u * kBuf;
            std::vector<double> got_l(kBuf);
            std::vector<double> got_r(kBuf);
            for (int j = 0; j < kBuf; ++j) {
                got_l[static_cast<size_t>(j)] = block[j];
                got_r[static_cast<size_t>(j)] = block[kBuf + j];
            }
            accumulate(got_l.data(), got_r.data(), y_l.data(), y_r.data(), kBuf, &intend_sig, &intend_noise);
            hrir_flip_buffer(true_l, flip_l, kFir);
            hrir_flip_buffer(true_r, flip_r, kFir);
            for (int i = 0; i < kFir; ++i) {
                h_l[static_cast<size_t>(i)] = flip_l[i];
                h_r[static_cast<size_t>(i)] = flip_r[i];
            }
            conv_ref_f64(ext.data(), h_l.data(), y_l.data(), kBuf, kFir);
            conv_ref_f64(ext.data(), h_r.data(), y_r.data(), kBuf, kFir);
            accumulate(got_l.data(), got_r.data(), y_l.data(), y_r.data(), kBuf, &all_sig, &all_noise);
            if (blended) {
                accumulate(got_l.data(), got_r.data(), y_l.data(), y_r.data(), kBuf, &blend_sig, &blend_noise);
                ++blend_n;
            }
        }
    }
    row->snr_intended = finish_snr(intend_sig, intend_noise);
    row->snr_all = finish_snr(all_sig, all_noise);
    row->blend_samples = blend_n;
    row->snr_blend = blend_n > 0 ? finish_snr(blend_sig, blend_noise) : 0.0;
}

int run_case(int sources, int mode, int hot, Row* row) {
    const int total = kWarmup + kMeasured;
    const int nx = kHist + total * kBuf;
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
    fprintf(stderr, "resident sources=%d mode=%d hot=%d\n", sources, mode, hot);
    if (motion_execute(x.data(), sources, total, kWarmup, kMeasured, hot, mode, 5, 0, y.data(), us.data(),
                       &uploads) != 0) {
        fprintf(stderr, "motion_execute failed\n");
        return 1;
    }
    row->sources = sources;
    row->kind = mode;
    row->hot = hot;
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
    fill_snr(x.data(), y.data(), sources, total, mode, hot, row);
    if (row->snr_intended < 100.0) {
        fprintf(stderr, "intended snr %.2f mode=%d hot=%d\n", row->snr_intended, mode, hot);
        return 1;
    }
    return 0;
}

const char* row_name(const Row& row) {
    if (row.kind == 0) {
        return "每段重送";
    }
    if (row.kind == 1) {
        return "整圈在显存";
    }
    return "只留 K";
}

const Row* find_row(const std::vector<Row>& rows, int sources, int mode, int hot) {
    for (const Row& row : rows) {
        if (row.sources == sources && row.kind == mode && (mode != 2 || row.hot == hot)) {
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
    const int source_sets[] = {32, 8};
    const int ks[] = {2, 4, 8, 16};
    std::vector<Row> rows;
    for (int sources : source_sets) {
        Row every{};
        if (run_case(sources, 0, 8, &every) != 0) {
            return 1;
        }
        rows.push_back(every);
        for (int hot : ks) {
            Row row{};
            if (run_case(sources, 2, hot, &row) != 0) {
                return 1;
            }
            rows.push_back(row);
        }
        Row ring{};
        if (run_case(sources, 1, 8, &ring) != 0) {
            return 1;
        }
        rows.push_back(ring);
    }

    int hot16[16];
    const int n16 = select_hot_grid(16, 5, hot16);
    const int cover = n16 > 0 ? hot16[n16 - 1] * 5 : 0;
    printf("# 只留 K 个方向\n\n");
    printf("设备上只放 K 套 5 度系数，不把整圈拷进显存。方位落在这 K 个方向中间才混合，外面整套从主机重送。\n");
    printf("K 从 0 度起连续放置。16 个方向覆盖到 %d 度。另两档是方向一变就重送，以及整圈网格都在显存。\n", cover);
    printf("32 路为主，8 路对照。块长 1024，预热 %d 段，计时 %d 段。\n\n", kWarmup, kMeasured);
    printf("| 路数 | 驻留 | p50 μs | p99 μs | 超时段 | 重送 | 全段 SNR dB | 混合段 SNR dB |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (const Row& row : rows) {
        printf("| %d | %s", row.sources, row_name(row));
        if (row.kind == 2) {
            printf(" %d", row.hot);
        }
        printf(" | %.2f | %.2f | %d | %ld | %.2f | ", row.p50, row.p99, row.underruns, row.reuploads, row.snr_all);
        if (row.blend_samples > 0) {
            printf("%.2f |\n", row.snr_blend);
        } else {
            printf("— |\n");
        }
    }
    printf("\n");
    printf("全段信噪比相对真实方位，重送的段用的就是真实系数，会把这个数字抬高。混合段只统计留在设备上按比例相加的段。\n");
    for (int sources : source_sets) {
        const Row* every = find_row(rows, sources, 0, 8);
        const Row* k2 = find_row(rows, sources, 2, 2);
        const Row* k16 = find_row(rows, sources, 2, 16);
        const Row* ring = find_row(rows, sources, 1, 8);
        if (every == nullptr || k2 == nullptr || k16 == nullptr || ring == nullptr) {
            continue;
        }
        printf("%d 路：每段重送 %ld 次、p50 %.2f μs；只留 2 个方向 %ld 次、%.2f μs；只留 16 个方向 %ld 次、%.2f μs；整圈 %ld 次、%.2f μs。\n",
               sources, every->reuploads, every->p50, k2->reuploads, k2->p50, k16->reuploads, k16->p50,
               ring->reuploads, ring->p50);
    }
    const Row* k16 = find_row(rows, 32, 2, 16);
    const Row* ring = find_row(rows, 32, 1, 8);
    const Row* best = nullptr;
    for (const Row& row : rows) {
        if (row.sources != 32) {
            continue;
        }
        if (best == nullptr || row.p50 < best->p50) {
            best = &row;
        }
    }
    if (k16 != nullptr && ring != nullptr && k16->reuploads > ring->reuploads) {
        printf("加到 16 个方向仍然比整圈多出重送，覆盖停在 %d 度，盖不住散在 0 度到 160 度的慢转。\n", cover);
    }
    if (best != nullptr) {
        printf("这次 32 路端到端较短的是%s", row_name(*best));
        if (best->kind == 2) {
            printf(" %d 个方向", best->hot);
        }
        printf("，p50 %.2f μs，重送 %ld 次。", best->p50, best->reuploads);
        if (best->kind == 1) {
            printf("划算的驻留是整圈留在显存，不是常量内存里的某个 K。\n");
        } else {
            printf("\n");
        }
    }
    printf("以上微秒是 WSL2 上的读数。\n");
    return 0;
}

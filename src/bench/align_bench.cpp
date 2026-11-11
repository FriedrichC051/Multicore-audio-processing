#include "conv.h"
#include "hrir_field.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kFir = 512;
constexpr int kBuf = 1024;
constexpr int kHist = kFir - 1;
constexpr int kWarmup = 20;
constexpr int kMeasured = 1000;
constexpr int kHot = 8;
constexpr double kFs = 48000.0;

enum Method { kLinear = 0, kAligned = 1, kNearest = 2 };

struct Score {
    int sources;
    int method;
    int samples;
    double snr;
    double itd_us;
    double ild_db;
};

double itd_us(const float* left, const float* right, int n) {
    constexpr int kLag = 40;
    double corr[2 * kLag + 1];
    for (int lag = -kLag; lag <= kLag; ++lag) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            const int j = i - lag;
            if (j < 0 || j >= n) {
                continue;
            }
            sum += static_cast<double>(left[i]) * right[j];
        }
        corr[lag + kLag] = sum;
    }
    int best = 0;
    for (int i = 1; i <= 2 * kLag; ++i) {
        if (corr[i] > corr[best]) {
            best = i;
        }
    }
    double delta = 0.0;
    if (best > 0 && best < 2 * kLag) {
        const double ym1 = corr[best - 1];
        const double y0 = corr[best];
        const double yp1 = corr[best + 1];
        const double denom = ym1 - 2.0 * y0 + yp1;
        if (denom != 0.0) {
            delta = 0.5 * (ym1 - yp1) / denom;
        }
        if (delta > 1.0) {
            delta = 1.0;
        }
        if (delta < -1.0) {
            delta = -1.0;
        }
    }
    const double lag = static_cast<double>(best - kLag) + delta;
    return lag * 1.0e6 / kFs;
}

double ild_db(const float* left, const float* right, int n) {
    double el = 0.0;
    double er = 0.0;
    for (int i = 0; i < n; ++i) {
        el += static_cast<double>(left[i]) * left[i];
        er += static_cast<double>(right[i]) * right[i];
    }
    return 10.0 * std::log10((el + 1e-20) / (er + 1e-20));
}

void make_hrir(int method, float az, const int* hot, int hot_n, float* left, float* right) {
    if (method == kLinear) {
        hrir_blend_linear(az, hot, hot_n, 5, left, right, kFir);
    } else if (method == kAligned) {
        hrir_blend_aligned(az, hot, hot_n, 5, left, right, kFir);
    } else {
        hrir_nearest(az, 15, left, right, kFir);
    }
}

void convolve(const double* ext, const float* left, const float* right, double* y_l, double* y_r) {
    float flip_l[kFir];
    float flip_r[kFir];
    std::vector<double> h_l(kFir);
    std::vector<double> h_r(kFir);
    hrir_flip_buffer(left, flip_l, kFir);
    hrir_flip_buffer(right, flip_r, kFir);
    for (int i = 0; i < kFir; ++i) {
        h_l[static_cast<size_t>(i)] = flip_l[i];
        h_r[static_cast<size_t>(i)] = flip_r[i];
    }
    conv_ref_f64(ext, h_l.data(), y_l, kBuf, kFir);
    conv_ref_f64(ext, h_r.data(), y_r, kBuf, kFir);
}

Score score_method(const float* x, int sources, int total, int method) {
    int hot[16];
    const int hot_n = select_hot_grid(kHot, 5, hot);
    std::vector<float> az(static_cast<size_t>(total) * sources);
    fill_azimuths(sources, total, kWarmup + 100, az.data());
    const int nx = kHist + total * kBuf;
    Score score{};
    score.sources = sources;
    score.method = method;
    double signal = 0.0;
    double noise = 0.0;
    double itd_abs = 0.0;
    double ild_abs = 0.0;
    int count = 0;
    std::vector<double> ext(static_cast<size_t>(kHist + kBuf));
    std::vector<double> y_l(kBuf);
    std::vector<double> y_r(kBuf);
    std::vector<double> t_l(kBuf);
    std::vector<double> t_r(kBuf);
    float approx_l[kFir];
    float approx_r[kFir];
    float true_l[kFir];
    float true_r[kFir];
    for (int pick = 0; pick < kMeasured; pick += 20) {
        const int b = kWarmup + pick;
        for (int s = 0; s < sources; ++s) {
            const float angle = az[static_cast<size_t>(b) * sources + s];
            TierHit hit{};
            classify_azimuth(angle, hot, hot_n, 5, &hit);
            if (hit.kind == kTierCold) {
                continue;
            }
            for (int i = 0; i < kHist + kBuf; ++i) {
                ext[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(s) * nx + static_cast<size_t>(b) * kBuf + i];
            }
            make_hrir(method, angle, hot, hot_n, approx_l, approx_r);
            hrir_causal(angle, true_l, true_r, kFir);
            convolve(ext.data(), approx_l, approx_r, y_l.data(), y_r.data());
            convolve(ext.data(), true_l, true_r, t_l.data(), t_r.data());
            for (int j = 0; j < kBuf; ++j) {
                const double dl = y_l[static_cast<size_t>(j)] - t_l[static_cast<size_t>(j)];
                const double dr = y_r[static_cast<size_t>(j)] - t_r[static_cast<size_t>(j)];
                signal += t_l[static_cast<size_t>(j)] * t_l[static_cast<size_t>(j)] +
                          t_r[static_cast<size_t>(j)] * t_r[static_cast<size_t>(j)];
                noise += dl * dl + dr * dr;
            }
            const double itd_err = itd_us(approx_l, approx_r, kFir) - itd_us(true_l, true_r, kFir);
            const double ild_err = ild_db(approx_l, approx_r, kFir) - ild_db(true_l, true_r, kFir);
            itd_abs += std::fabs(itd_err);
            ild_abs += std::fabs(ild_err);
            ++count;
        }
    }
    score.samples = count;
    score.snr = noise == 0.0 ? 200.0 : 10.0 * std::log10(signal / noise);
    score.itd_us = count > 0 ? itd_abs / count : 0.0;
    score.ild_db = count > 0 ? ild_abs / count : 0.0;
    return score;
}

const char* method_name(int method) {
    if (method == kLinear) {
        return "5度按比例相加";
    }
    if (method == kAligned) {
        return "延迟对齐后只混幅度";
    }
    return "15度最近邻";
}

const Score* find_score(const std::vector<Score>& rows, int sources, int method) {
    for (const Score& row : rows) {
        if (row.sources == sources && row.method == method) {
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
    const int source_sets[] = {8, 32};
    const int methods[] = {kLinear, kAligned, kNearest};
    std::vector<Score> rows;
    for (int sources : source_sets) {
        const int total = kWarmup + kMeasured;
        const int nx = kHist + total * kBuf;
        std::vector<float> x(static_cast<size_t>(sources) * nx);
        float unused_l[kFir];
        float unused_r[kFir];
        for (int s = 0; s < sources; ++s) {
            signal_generate_at(x.data() + static_cast<size_t>(s) * nx, nx, kHist, unused_l, unused_r, kFir,
                               0x00C0FFEEu + static_cast<uint32_t>(s) * 17u);
        }
        for (int method : methods) {
            fprintf(stderr, "align sources=%d method=%d\n", sources, method);
            rows.push_back(score_method(x.data(), sources, total, method));
        }
    }

    printf("# 延迟对齐与最近邻\n\n");
    printf("5 度网格、热方向 8 个，运动与主实验相同。三列都只统计热区和温区命中段，相对该方位的真实系数。\n");
    printf("每隔 20 段取一次。15 度最近邻用的是同一批命中段，不做插值。\n\n");
    printf("| 路数 | 做法 | 命中样本 | SNR dB | ITD 误差 μs | ILD 误差 dB |\n");
    printf("| --- | --- | --- | --- | --- | --- |\n");
    for (const Score& row : rows) {
        printf("| %d | %s | %d | %.2f | %.2f | %.3f |\n", row.sources, method_name(row.method), row.samples,
               row.snr, row.itd_us, row.ild_db);
    }
    printf("\n");
    for (int sources : source_sets) {
        const Score* linear = find_score(rows, sources, kLinear);
        const Score* aligned = find_score(rows, sources, kAligned);
        const Score* nearest = find_score(rows, sources, kNearest);
        if (linear == nullptr || aligned == nullptr || nearest == nullptr) {
            continue;
        }
        printf("%d 路：按比例相加 %.2f dB，对齐后 %.2f dB，最近邻 %.2f dB。", sources, linear->snr, aligned->snr,
               nearest->snr);
        if (aligned->snr >= linear->snr + 3.0) {
            printf("对齐后信噪比明显升高，延迟跳变是按比例相加的主要误差，这一点得到验证。\n");
        } else if (aligned->snr + 1.0 < linear->snr) {
            printf("对齐后信噪比下降，不能把误差只归因于延迟跳变，还有别的误差。\n");
        } else {
            printf("对齐后信噪比没有明显升高，还有别的误差。\n");
        }
        printf("双耳时间差的平均绝对误差：按比例相加 %.2f μs，对齐后 %.2f μs，最近邻 %.2f μs。", linear->itd_us,
               aligned->itd_us, nearest->itd_us);
        printf("声级差：%.3f dB、%.3f dB、%.3f dB。\n", linear->ild_db, aligned->ild_db, nearest->ild_db);
    }
    printf("大约 10 微秒只是听感文献里的一个刻度。这套系数不是人头测量，没有做试听，不能说听起来合格。\n");
    return 0;
}

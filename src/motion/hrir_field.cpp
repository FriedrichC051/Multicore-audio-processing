#include "hrir_field.hpp"

#include <cmath>
#include <cstring>

namespace {

float wrap360(float deg) {
    float x = std::fmod(deg, 360.0f);
    if (x < 0.0f) {
        x += 360.0f;
    }
    return x;
}

bool grid_is_hot(int grid, const int* hot_grid, int hot_count) {
    for (int i = 0; i < hot_count; ++i) {
        if (hot_grid[i] == grid) {
            return true;
        }
    }
    return false;
}

}  // namespace

bool azimuth_in_cold_gap(float azimuth_deg) {
    const float az = wrap360(azimuth_deg);
    return az >= 170.0f && az < 190.0f;
}

void hrir_causal(float azimuth_deg, float* left, float* right, int fir) {
    const float s = std::sin(wrap360(azimuth_deg) * 3.14159265f / 180.0f);
    const float amp_l = 0.35f * (1.0f + 0.6f * s);
    const float amp_r = 0.35f * (1.0f - 0.6f * s);
    int delay_l = 20 + static_cast<int>(std::lround(12.0f * s));
    int delay_r = 20 + static_cast<int>(std::lround(-12.0f * s));
    if (delay_l < 0) {
        delay_l = 0;
    }
    if (delay_r < 0) {
        delay_r = 0;
    }
    if (delay_l >= fir) {
        delay_l = fir - 1;
    }
    if (delay_r >= fir) {
        delay_r = fir - 1;
    }
    std::memset(left, 0, static_cast<size_t>(fir) * sizeof(float));
    std::memset(right, 0, static_cast<size_t>(fir) * sizeof(float));
    for (int i = delay_l; i < fir; ++i) {
        left[i] = amp_l * std::exp(-(i - delay_l) / 40.0f);
    }
    for (int i = delay_r; i < fir; ++i) {
        right[i] = amp_r * std::exp(-(i - delay_r) / 40.0f);
    }
}

void hrir_flip_buffer(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = src[n - 1 - i];
    }
}

void classify_azimuth(float azimuth_deg, const int* hot_grid, int hot_count, TierHit* hit) {
    const float az = wrap360(azimuth_deg);
    if (azimuth_in_cold_gap(az)) {
        hit->kind = kTierCold;
        hit->index_a = 0;
        hit->index_b = 0;
        hit->weight_b = 0.0f;
        return;
    }
    int i0 = static_cast<int>(std::floor(az / static_cast<float>(kGridStepDeg)));
    if (i0 >= kGridCount) {
        i0 = kGridCount - 1;
    }
    int i1 = (i0 + 1) % kGridCount;
    const float base = static_cast<float>(i0 * kGridStepDeg);
    float w = (az - base) / static_cast<float>(kGridStepDeg);
    if (w < 0.0f) {
        w = 0.0f;
    }
    if (w > 1.0f) {
        w = 1.0f;
    }
    if (azimuth_in_cold_gap(static_cast<float>(i0 * kGridStepDeg)) ||
        azimuth_in_cold_gap(static_cast<float>(i1 * kGridStepDeg))) {
        hit->kind = kTierCold;
        hit->index_a = i0;
        hit->index_b = i1;
        hit->weight_b = w;
        return;
    }
    hit->index_a = i0;
    hit->index_b = i1;
    hit->weight_b = w;
    const bool both_hot = grid_is_hot(i0, hot_grid, hot_count) && grid_is_hot(i1, hot_grid, hot_count);
    hit->kind = both_hot ? kTierHot : kTierWarm;
}

int select_hot_grid(int hot_count, int* hot_grid) {
    int warm[kGridCount];
    int n_warm = 0;
    for (int i = 0; i < kGridCount; ++i) {
        if (!azimuth_in_cold_gap(static_cast<float>(i * kGridStepDeg))) {
            warm[n_warm++] = i;
        }
    }
    int n = 0;
    for (int i = 0; i < n_warm && n < hot_count; ++i) {
        if (warm[i] * kGridStepDeg <= 160) {
            hot_grid[n++] = warm[i];
        }
    }
    return n;
}

void fill_azimuths(int sources, int total_blocks, int jump_block, float* azimuths) {
    float angle[64];
    int dir[64];
    const int n = sources < 64 ? sources : 64;
    for (int s = 0; s < n; ++s) {
        angle[s] = static_cast<float>((10 * s) % 150);
        dir[s] = 1;
    }
    if (n > 0) {
        angle[0] = 0.0f;
    }
    for (int b = 0; b < total_blocks; ++b) {
        for (int s = 0; s < n; ++s) {
            const bool jumper = n > 1 && s >= n - 2;
            const bool still = n > 1 && s == 0;
            if (still) {
                azimuths[static_cast<size_t>(b) * sources + s] = 0.0f;
                continue;
            }
            if (jumper && b >= jump_block) {
                azimuths[static_cast<size_t>(b) * sources + s] = 175.0f;
                continue;
            }
            angle[s] += static_cast<float>(dir[s]);
            if (angle[s] > 160.0f) {
                angle[s] = 160.0f;
                dir[s] = -1;
            }
            if (angle[s] < 0.0f) {
                angle[s] = 0.0f;
                dir[s] = 1;
            }
            azimuths[static_cast<size_t>(b) * sources + s] = angle[s];
        }
    }
}

void reference_flipped(float azimuth_deg, const int* hot_grid, int hot_count, int reupload_mode, float* dst,
                       int fir) {
    TierHit hit{};
    classify_azimuth(azimuth_deg, hot_grid, hot_count, &hit);
    float left[kFirDefault];
    float right[kFirDefault];
    float other_l[kFirDefault];
    float other_r[kFirDefault];
    float mix_l[kFirDefault];
    float mix_r[kFirDefault];
    if (fir > kFirDefault) {
        return;
    }
    if (reupload_mode != 0 || hit.kind == kTierCold) {
        hrir_causal(azimuth_deg, left, right, fir);
    } else {
        hrir_causal(static_cast<float>(hit.index_a * kGridStepDeg), left, right, fir);
        hrir_causal(static_cast<float>(hit.index_b * kGridStepDeg), other_l, other_r, fir);
        const float w = hit.weight_b;
        for (int i = 0; i < fir; ++i) {
            mix_l[i] = (1.0f - w) * left[i] + w * other_l[i];
            mix_r[i] = (1.0f - w) * right[i] + w * other_r[i];
        }
        std::memcpy(left, mix_l, static_cast<size_t>(fir) * sizeof(float));
        std::memcpy(right, mix_r, static_cast<size_t>(fir) * sizeof(float));
    }
    hrir_flip_buffer(left, dst, fir);
    hrir_flip_buffer(right, dst + fir, fir);
}

int hrir_field_self_check() {
    float a_l[32];
    float a_r[32];
    float b_l[32];
    float b_r[32];
    hrir_causal(0.0f, a_l, a_r, 32);
    hrir_causal(0.0f, b_l, b_r, 32);
    for (int i = 0; i < 32; ++i) {
        if (a_l[i] != b_l[i] || a_r[i] != b_r[i]) {
            return 1;
        }
    }
    hrir_causal(90.0f, b_l, b_r, 32);
    int same = 1;
    for (int i = 0; i < 32; ++i) {
        if (b_l[i] != a_l[i] || b_r[i] != a_r[i]) {
            same = 0;
        }
    }
    if (same) {
        return 1;
    }
    int hot[4];
    if (select_hot_grid(4, hot) != 4) {
        return 1;
    }
    TierHit hit{};
    classify_azimuth(0.0f, hot, 4, &hit);
    if (hit.kind == kTierCold) {
        return 1;
    }
    classify_azimuth(175.0f, hot, 4, &hit);
    if (hit.kind != kTierCold) {
        return 1;
    }
    float exact[64];
    float blended[64];
    reference_flipped(0.0f, hot, 4, 1, exact, 32);
    reference_flipped(0.0f, hot, 4, 0, blended, 32);
    for (int i = 0; i < 64; ++i) {
        if (exact[i] != blended[i]) {
            return 1;
        }
    }
    reference_flipped(2.5f, hot, 4, 1, exact, 32);
    reference_flipped(2.5f, hot, 4, 0, blended, 32);
    int differ = 0;
    for (int i = 0; i < 64; ++i) {
        if (exact[i] != blended[i]) {
            differ = 1;
        }
    }
    if (!differ) {
        return 1;
    }
    return 0;
}

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

int grid_count_for(int grid_step) {
    if (grid_step != 5 && grid_step != 15 && grid_step != 30) {
        return 0;
    }
    return 360 / grid_step;
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

void classify_azimuth(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, TierHit* hit) {
    const float az = wrap360(azimuth_deg);
    const int count = grid_count_for(grid_step);
    if (count == 0 || azimuth_in_cold_gap(az)) {
        hit->kind = kTierCold;
        hit->index_a = 0;
        hit->index_b = 0;
        hit->weight_b = 0.0f;
        return;
    }
    int i0 = static_cast<int>(std::floor(az / static_cast<float>(grid_step)));
    if (i0 >= count) {
        i0 = count - 1;
    }
    if (i0 < 0) {
        i0 = 0;
    }
    int i1 = (i0 + 1) % count;
    const float base = static_cast<float>(i0 * grid_step);
    float w = (az - base) / static_cast<float>(grid_step);
    if (w < 0.0f) {
        w = 0.0f;
    }
    if (w > 1.0f) {
        w = 1.0f;
    }
    if (azimuth_in_cold_gap(static_cast<float>(i0 * grid_step)) ||
        azimuth_in_cold_gap(static_cast<float>(i1 * grid_step))) {
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

int select_hot_grid(int hot_count, int grid_step, int* hot_grid) {
    const int count = grid_count_for(grid_step);
    int stored[kGridMax];
    int n_stored = 0;
    for (int i = 0; i < count; ++i) {
        if (!azimuth_in_cold_gap(static_cast<float>(i * grid_step))) {
            stored[n_stored++] = i;
        }
    }
    int n = 0;
    for (int i = 0; i < n_stored && n < hot_count; ++i) {
        if (stored[i] * grid_step <= 160) {
            hot_grid[n++] = stored[i];
        }
    }
    for (int i = 0; i < n_stored && n < hot_count; ++i) {
        if (stored[i] * grid_step > 160) {
            hot_grid[n++] = stored[i];
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

void grid_endpoints(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, int force_exact,
                    float* left_a, float* right_a, float* left_b, float* right_b, float* weight_b, int* kind,
                    int fir) {
    TierHit hit{};
    classify_azimuth(azimuth_deg, hot_grid, hot_count, grid_step, &hit);
    if (fir > kFirDefault) {
        return;
    }
    if (force_exact != 0 || hit.kind == kTierCold) {
        hrir_causal(azimuth_deg, left_a, right_a, fir);
        std::memcpy(left_b, left_a, static_cast<size_t>(fir) * sizeof(float));
        std::memcpy(right_b, right_a, static_cast<size_t>(fir) * sizeof(float));
        *weight_b = 0.0f;
        *kind = kTierCold;
        return;
    }
    hrir_causal(static_cast<float>(hit.index_a * grid_step), left_a, right_a, fir);
    hrir_causal(static_cast<float>(hit.index_b * grid_step), left_b, right_b, fir);
    *weight_b = hit.weight_b;
    *kind = hit.kind;
}

void reference_flipped(float azimuth_deg, const int* hot_grid, int hot_count, int reupload_mode, int grid_step,
                       float* dst, int fir) {
    float left[kFirDefault];
    float right[kFirDefault];
    float other_l[kFirDefault];
    float other_r[kFirDefault];
    float mix_l[kFirDefault];
    float mix_r[kFirDefault];
    float weight = 0.0f;
    int kind = kTierCold;
    if (fir > kFirDefault) {
        return;
    }
    grid_endpoints(azimuth_deg, hot_grid, hot_count, grid_step, reupload_mode, left, right, other_l, other_r,
                   &weight, &kind, fir);
    if (kind != kTierCold) {
        for (int i = 0; i < fir; ++i) {
            mix_l[i] = (1.0f - weight) * left[i] + weight * other_l[i];
            mix_r[i] = (1.0f - weight) * right[i] + weight * other_r[i];
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
    int hot[8];
    if (select_hot_grid(4, 5, hot) != 4) {
        return 1;
    }
    TierHit hit{};
    classify_azimuth(0.0f, hot, 4, 5, &hit);
    if (hit.kind == kTierCold) {
        return 1;
    }
    classify_azimuth(175.0f, hot, 4, 5, &hit);
    if (hit.kind != kTierCold) {
        return 1;
    }
    if (select_hot_grid(8, 5, hot) != 8 || select_hot_grid(8, 15, hot) != 8 ||
        select_hot_grid(8, 30, hot) != 8) {
        return 1;
    }
    classify_azimuth(160.0f, hot, 8, 5, &hit);
    if (hit.kind == kTierCold) {
        return 1;
    }
    classify_azimuth(160.0f, hot, 8, 15, &hit);
    if (hit.kind == kTierCold) {
        return 1;
    }
    classify_azimuth(160.0f, hot, 8, 30, &hit);
    if (hit.kind != kTierCold) {
        return 1;
    }
    float exact[64];
    float blended[64];
    reference_flipped(0.0f, hot, 4, 1, 5, exact, 32);
    reference_flipped(0.0f, hot, 4, 0, 5, blended, 32);
    for (int i = 0; i < 64; ++i) {
        if (exact[i] != blended[i]) {
            return 1;
        }
    }
    reference_flipped(2.5f, hot, 4, 1, 5, exact, 32);
    reference_flipped(2.5f, hot, 4, 0, 5, blended, 32);
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

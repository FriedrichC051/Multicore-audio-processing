#ifndef HRIR_FIELD_HPP
#define HRIR_FIELD_HPP

// Horizontal ring of smooth head-related filters and the hot/warm/cold decision.
// A direction is cold when it falls in the gap that is not stored on the device.

enum { kGridStepDeg = 5, kGridCount = 360 / kGridStepDeg, kFirDefault = 512 };

enum TierKind { kTierHot = 0, kTierWarm = 1, kTierCold = 2 };

struct TierHit {
    int kind;
    int index_a;
    int index_b;
    float weight_b;
};

void hrir_causal(float azimuth_deg, float* left, float* right, int fir);
void hrir_flip_buffer(const float* src, float* dst, int n);

bool azimuth_in_cold_gap(float azimuth_deg);
void classify_azimuth(float azimuth_deg, const int* hot_grid, int hot_count, TierHit* hit);

int select_hot_grid(int hot_count, int* hot_grid);

void fill_azimuths(int sources, int total_blocks, int jump_block, float* azimuths);

void reference_flipped(float azimuth_deg, const int* hot_grid, int hot_count, int reupload_mode, float* dst,
                       int fir);

int hrir_field_self_check();

#endif

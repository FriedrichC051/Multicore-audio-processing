#ifndef HRIR_FIELD_HPP
#define HRIR_FIELD_HPP

// Horizontal ring of smooth head-related filters and the hot/warm/cold decision.
// A direction is cold when it falls in the gap that is not stored on the device.

enum { kGridStepDeg = 5, kGridCount = 360 / kGridStepDeg, kGridMax = 72, kFirDefault = 512 };

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
int grid_count_for(int grid_step);
void classify_azimuth(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, TierHit* hit);

int select_hot_grid(int hot_count, int grid_step, int* hot_grid);

void fill_azimuths(int sources, int total_blocks, int jump_block, float* azimuths);

void grid_endpoints(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, int force_exact,
                    float* left_a, float* right_a, float* left_b, float* right_b, float* weight_b, int* kind,
                    int fir);

void reference_flipped(float azimuth_deg, const int* hot_grid, int hot_count, int reupload_mode, int grid_step,
                       float* dst, int fir);

void hrir_blend_linear(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, float* left,
                       float* right, int fir);
void hrir_blend_aligned(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, float* left,
                        float* right, int fir);
void hrir_nearest(float azimuth_deg, int grid_step, float* left, float* right, int fir);

// 1 when azimuth lies between the retained directions, inclusive. 0 means the host must send the true set.
int resident_bracket(float azimuth_deg, const int* hot_grid, int hot_count, int grid_step, TierHit* hit);

int hrir_field_self_check();

#endif

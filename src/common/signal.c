#include "conv.h"

#include <math.h>

int conv_total_samples(int warmup, int blocks) {
    return CONV_HIST + (warmup + blocks) * CONV_BUF;
}

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float lcg_signed(uint32_t *state) {
    return (lcg_next(state) * (1.0f / 2147483648.0f)) - 1.0f;
}

void hrir_generate(float *h_l, float *h_r, int fir, uint32_t seed) {
    uint32_t sl = seed ^ 0xA5A5A5A5u;
    uint32_t sr = seed ^ 0x5A5A5A5Au;
    int i;
    for (i = 0; i < fir; ++i) {
        float env = expf(-(float)i / 64.0f);
        h_l[i] = 0.2f * env * lcg_signed(&sl);
        h_r[i] = 0.2f * env * lcg_signed(&sr);
    }
}

void signal_generate_at(float *x, int nx, int hist, float *h_l, float *h_r, int fir, uint32_t seed) {
    uint32_t sx = seed;
    int i;

    if (hist > nx) {
        hist = nx;
    }
    for (i = 0; i < hist; ++i) {
        x[i] = 0.0f;
    }
    for (; i < nx; ++i) {
        x[i] = 0.2f * lcg_signed(&sx);
    }
    hrir_generate(h_l, h_r, fir, seed);
}

void signal_generate(float *x, int nx, float *h_l, float *h_r, int fir) {
    signal_generate_at(x, nx, CONV_HIST, h_l, h_r, fir, CONV_SEED);
}

void hrir_flip(const float *h, float *h_flip, int fir) {
    int k;
    for (k = 0; k < fir; ++k) {
        h_flip[k] = h[fir - 1 - k];
    }
}

int16_t conv_to_q15(float x) {
    float scaled = nearbyintf(x * 32768.0f);
    if (scaled > 32767.0f) {
        scaled = 32767.0f;
    }
    if (scaled < -32768.0f) {
        scaled = -32768.0f;
    }
    return (int16_t)scaled;
}

int8_t conv_to_i8(float x) {
    float scaled = nearbyintf(x * (float)CONV_I8_SCALE);
    if (scaled > 127.0f) {
        scaled = 127.0f;
    }
    if (scaled < -127.0f) {
        scaled = -127.0f;
    }
    return (int8_t)scaled;
}

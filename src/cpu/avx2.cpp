#include "conv.h"

#include <immintrin.h>

static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static float dot_scalar_tail(const float *x, const float *h, int begin, int fir) {
    float acc = 0.0f;
    int k;
    for (k = begin; k < fir; ++k) {
        acc += x[k] * h[k];
    }
    return acc;
}

void conv_avx2(const float *ext, const float *h_flip, float *y, int buf, int fir) {
    int j = 0;
    for (; j + 3 < buf; j += 4) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 7 < fir; k += 8) {
            __m256 hv = _mm256_loadu_ps(h_flip + k);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ext + j + k), hv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ext + j + 1 + k), hv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(ext + j + 2 + k), hv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(ext + j + 3 + k), hv, a3);
        }
        y[j] = hsum256(a0) + dot_scalar_tail(ext + j, h_flip, k, fir);
        y[j + 1] = hsum256(a1) + dot_scalar_tail(ext + j + 1, h_flip, k, fir);
        y[j + 2] = hsum256(a2) + dot_scalar_tail(ext + j + 2, h_flip, k, fir);
        y[j + 3] = hsum256(a3) + dot_scalar_tail(ext + j + 3, h_flip, k, fir);
    }
    for (; j < buf; ++j) {
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 7 < fir; k += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ext + j + k), _mm256_loadu_ps(h_flip + k), acc);
        }
        y[j] = hsum256(acc) + dot_scalar_tail(ext + j, h_flip, k, fir);
    }
}

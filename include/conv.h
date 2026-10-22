#ifndef CONV_H
#define CONV_H

#include <stddef.h>
#include <stdint.h>

#define CONV_BUF 1024
#define CONV_FIR 512
#define CONV_HIST (CONV_FIR - 1)
#define CONV_FS 48000
#define CONV_SEED 0x00C0FFEEu
#define CONV_WARMUP 20
#define CONV_BLOCKS 1000
#define CONV_I8_SCALE 127

int conv_total_samples(int warmup, int blocks);

void signal_generate(float *x, int nx, float *h_l, float *h_r, int fir);
void signal_generate_at(float *x, int nx, int hist, float *h_l, float *h_r, int fir, uint32_t seed);
void hrir_generate(float *h_l, float *h_r, int fir, uint32_t seed);
void hrir_flip(const float *h, float *h_flip, int fir);

int16_t conv_to_q15(float x);
int8_t conv_to_i8(float x);

void conv_ref_f64(const double *ext, const double *h_flip, double *y, int buf, int fir);
int conv_ref_self_check(void);

void conv_q15_i32(const int16_t *ext, const int16_t *h_flip, int16_t *y, int buf, int fir);
void conv_q15_i64(const int16_t *ext, const int16_t *h_flip, int16_t *y, int buf, int fir);
long conv_q15_i32_overflows(const int16_t *x, const int16_t *h_l, const int16_t *h_r,
                            int abs_begin, int blocks);
int conv_q15_overflow_self_check(void);

void conv_avx2(const float *ext, const float *h_flip, float *y, int buf, int fir);

typedef struct {
    double p50_us;
    double p99_us;
    double max_us;
} TimeStats;

double snr_db(const double *ref_l, const double *test_l, const double *ref_r, const double *test_r,
              int n);
void time_percentiles(double *us, int n, TimeStats *out);

#endif

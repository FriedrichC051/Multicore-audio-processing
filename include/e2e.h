#ifndef E2E_H
#define E2E_H

#ifdef __cplusplus
extern "C" {
#endif

/* x: [source][nx], nx = hist + total_blocks * buf, hist = fir - 1.
   The first hist samples of each source are the initial history.
   h_flip: [orient][source][2][fir], already reversed.
   y_out: [measured_block][source][2][buf]. */
int e2e_time_measure(int buf, int fir, int sources, int depth, int swap, int norient,
                     const float *x, const float *h_flip, int total_blocks, int warmup,
                     int measured, float *y_out, double *us_e2e, double *us_kernel);

/* h_orig: [2][fir], causal (not flipped). One source.
   y_out is measured audio blocks in order, each buf frames, stereo planar.
   us_* are per FFT group, without the (k-1) block algorithmic wait. */
int e2e_fft_measure(int buf, int fir, int k, const float *x, const float *h_orig, int total_blocks,
                    int warmup, int measured, float *y_out, double *us_e2e, double *us_compute,
                    int *groups_out);

void e2e_fft_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif

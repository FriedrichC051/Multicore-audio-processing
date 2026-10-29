#ifndef PATHS_H
#define PATHS_H

void path_enable_mapped_host(void);

/* Windowed paths keep history on the host. y_out: [block][source][2][buf].
   Returns 0 on success, 2 if this path is unavailable in the current environment, 1 on error.
   us_* have `measured` entries. bytes_per_block is the explicit cudaMemcpy traffic. */
int path_pageable_measure(int buf, int fir, int sources, const float *x, const float *h_flip,
                          int total_blocks, int warmup, int measured, float *y_out, double *us_e2e,
                          double *us_kernel, double *us_copy, long *bytes_per_block);

int path_async_measure(int buf, int fir, int sources, const float *x, const float *h_flip,
                       int total_blocks, int warmup, int measured, float *y_out, double *us_e2e,
                       double *us_kernel, double *us_copy, long *bytes_per_block);

int path_mapped_measure(int buf, int fir, int sources, const float *x, const float *h_flip,
                        int total_blocks, int warmup, int measured, float *y_out, double *us_e2e,
                        double *us_kernel, double *us_copy, long *bytes_per_block);

/* Returns 0 when a platform and a device exist. */
int opencl_probe(char *message, int message_bytes);

int opencl_resident_measure(int buf, int fir, int sources, const float *x, const float *h_flip,
                            int total_blocks, int warmup, int measured, float *y_out, double *us_e2e,
                            double *us_kernel, double *us_copy, long *bytes_per_block);

#endif

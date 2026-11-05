#ifndef MOTION_HPP
#define MOTION_HPP

// mode 0: reupload a coefficient set from the host whenever that source's direction changes.
// mode 1: hot blend from constant memory, warm blend from global memory, cold from the host.
int motion_execute(const float* x, int sources, int total_blocks, int warmup, int measured, int hot_count,
                   int mode, float* y_out, double* us_e2e, long* host_reuploads);

int motion_hot_limit();

#endif

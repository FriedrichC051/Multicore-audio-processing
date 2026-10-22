#include "conv.h"

#include <math.h>
#include <stdlib.h>

double snr_db(const double *ref_l, const double *test_l, const double *ref_r, const double *test_r,
              int n) {
    double signal = 0.0;
    double noise = 0.0;
    int i;

    for (i = 0; i < n; ++i) {
        double dl = test_l[i] - ref_l[i];
        double dr = test_r[i] - ref_r[i];
        signal += ref_l[i] * ref_l[i] + ref_r[i] * ref_r[i];
        noise += dl * dl + dr * dr;
    }
    if (noise == 0.0) {
        return INFINITY;
    }
    if (signal == 0.0) {
        return -INFINITY;
    }
    return 10.0 * log10(signal / noise);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) {
        return -1;
    }
    if (da > db) {
        return 1;
    }
    return 0;
}

static int rank_index(int n, double fraction) {
    int index = (int)ceil(fraction * (double)n) - 1;
    if (index < 0) {
        index = 0;
    }
    if (index >= n) {
        index = n - 1;
    }
    return index;
}

void time_percentiles(double *us, int n, TimeStats *out) {
    qsort(us, (size_t)n, sizeof(double), cmp_double);
    out->p50_us = us[rank_index(n, 0.50)];
    out->p99_us = us[rank_index(n, 0.99)];
    out->max_us = us[n - 1];
}

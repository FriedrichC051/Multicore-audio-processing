#include "conv.h"

#include <math.h>

void conv_ref_f64(const double *ext, const double *h_flip, double *y, int buf, int fir) {
    int j;
    for (j = 0; j < buf; ++j) {
        const double *x = ext + j;
        double acc = 0.0;
        int k;
        for (k = 0; k < fir; ++k) {
            acc += x[k] * h_flip[k];
        }
        y[j] = acc;
    }
}

int conv_ref_self_check(void) {
    /* hist = 3, buf = 4, fir = 4. Impulse at the first new sample.
       h = {1,2,3,4}, h_flip = {4,3,2,1}. */
    double ext[7] = {0, 0, 0, 1, 0, 0, 0};
    double h_flip[4] = {4, 3, 2, 1};
    double y[4];
    double expect[4] = {1, 2, 3, 4};
    int j;

    conv_ref_f64(ext, h_flip, y, 4, 4);
    for (j = 0; j < 4; ++j) {
        if (fabs(y[j] - expect[j]) > 1e-12) {
            return 1;
        }
    }
    return 0;
}

#include "lib.hpp"

void conv1_feature_extract(
    data_t a[IMG_SIZE],
    const weight_t weights[CONV1_OUT_CH * IN_CH * K * K],
    const weight_t bias[CONV1_OUT_CH],
    data_t conv1_out[CONV1_SIZE]
) {
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        for (int y = 0; y < CONV1_H; y++) {
            for (int x = 0; x < CONV1_W; x++) {
#pragma HLS PIPELINE
                acc_t sum = bias[oc];

                for (int ky = 0; ky < K; ky++) {
                    for (int kx = 0; kx < K; kx++) {
#pragma HLS UNROLL
                        int iy = y + ky - PAD;
                        int ix = x + kx - PAD;

                        if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W) {
                            int img_idx = iy * IMG_W + ix;
                            int w_idx = oc * IN_CH * K * K + ky * K + kx;
                            sum += (acc_t)a[img_idx] * (acc_t)weights[w_idx];
                        }
                    }
                }

                conv1_out[(oc * CONV1_H + y) * CONV1_W + x] = (data_t)sum;
            }
        }
    }
}

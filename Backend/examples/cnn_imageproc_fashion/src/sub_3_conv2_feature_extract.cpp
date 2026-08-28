#include "lib.hpp"

void conv2_feature_extract(
    data_t pool1_out[POOL1_SIZE],
    const weight_t weights[CONV2_OUT_CH * CONV1_OUT_CH * K * K],
    const weight_t bias[CONV2_OUT_CH],
    data_t conv2_out[CONV2_SIZE]
) {
#pragma HLS LOOP_FLATTEN off
    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
        for (int y = 0; y < CONV2_H; y++) {
            for (int x = 0; x < CONV2_W; x++) {
                acc_t sum = bias[oc];

                for (int ic = 0; ic < CONV1_OUT_CH; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            int iy = y + ky - PAD;
                            int ix = x + kx - PAD;

                            if (iy >= 0 && iy < POOL1_H && ix >= 0 && ix < POOL1_W) {
                                int in_idx = (ic * POOL1_H + iy) * POOL1_W + ix;
                                int w_idx = ((oc * CONV1_OUT_CH + ic) * K + ky) * K + kx;
                                sum += (acc_t)pool1_out[in_idx] * (acc_t)weights[w_idx];
                            }
                        }
                    }
                }

                conv2_out[(oc * CONV2_H + y) * CONV2_W + x] = (data_t)sum;
            }
        }
    }
}

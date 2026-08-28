#include "lib.hpp"

void relu_pool2(
    data_t conv2_out[CONV2_SIZE],
    data_t pool2_out[POOL2_SIZE]
) {
    for (int c = 0; c < CONV2_OUT_CH; c++) {
        for (int y = 0; y < POOL2_H; y++) {
            for (int x = 0; x < POOL2_W; x++) {
#pragma HLS PIPELINE
                data_t max_val = 0;

                for (int py = 0; py < 2; py++) {
                    for (int px = 0; px < 2; px++) {
#pragma HLS UNROLL
                        int iy = y * 2 + py;
                        int ix = x * 2 + px;
                        int idx = (c * CONV2_H + iy) * CONV2_W + ix;
                        data_t val = conv2_out[idx];

                        if (val < 0) val = 0;
                        if (py == 0 && px == 0) max_val = val;
                        else if (val > max_val) max_val = val;
                    }
                }

                pool2_out[(c * POOL2_H + y) * POOL2_W + x] = max_val;
            }
        }
    }
}

#include "lib.hpp"

void relu_pool1(
    data_t conv1_out[CONV1_SIZE],
    data_t pool1_out[POOL1_SIZE]
) {
    for (int c = 0; c < CONV1_OUT_CH; c++) {
        for (int y = 0; y < POOL1_H; y++) {
            for (int x = 0; x < POOL1_W; x++) {
#pragma HLS PIPELINE
                data_t max_val = 0;

                for (int py = 0; py < 2; py++) {
                    for (int px = 0; px < 2; px++) {
#pragma HLS UNROLL
                        int iy = y * 2 + py;
                        int ix = x * 2 + px;
                        int idx = (c * CONV1_H + iy) * CONV1_W + ix;
                        data_t val = conv1_out[idx];

                        if (val < 0) val = 0;
                        if (py == 0 && px == 0) max_val = val;
                        else if (val > max_val) max_val = val;
                    }
                }

                pool1_out[(c * POOL1_H + y) * POOL1_W + x] = max_val;
            }
        }
    }
}

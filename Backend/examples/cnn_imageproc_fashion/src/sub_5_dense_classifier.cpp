#include "lib.hpp"

void dense_classifier(
    data_t pool2_out[POOL2_SIZE],
    const weight_t weights[NUM_CLASSES * DENSE_IN_SIZE],
    const weight_t bias[NUM_CLASSES],
    data_t b[NUM_CLASSES]
) {
    const int PARALLEL = 8;

    for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10
        acc_t partial[PARALLEL];
#pragma HLS ARRAY_PARTITION variable=partial complete

        for (int p = 0; p < PARALLEL; p++) {
#pragma HLS UNROLL
            partial[p] = 0;
        }

        for (int i = 0; i < DENSE_IN_SIZE; i += PARALLEL) {
#pragma HLS PIPELINE II=1
            for (int p = 0; p < PARALLEL; p++) {
#pragma HLS UNROLL
                partial[p] +=
                    (acc_t)pool2_out[i + p] *
                    (acc_t)weights[c * DENSE_IN_SIZE + i + p];
            }
        }

        acc_t sum = bias[c];
        for (int p = 0; p < PARALLEL; p++) {
#pragma HLS UNROLL
            sum += partial[p];
        }

        b[c] = (data_t)sum;
    }
}

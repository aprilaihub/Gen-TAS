#include "lib.hpp"
#include "weights.hpp"

void array_product(
    const histogram_t histogram[CAMC_HISTOGRAM_SIZE],
    score_t result[CAMC_SCORE_COUNT]
) {
    score_t scores[10] = {0};

#define ACCUMULATE(SCORE, PREFIX) \
    for (int i = 0; i < Length_##PREFIX##_10; ++i) { \
        _Pragma("HLS PIPELINE") \
        const int index = (Lite_##PREFIX##_weight_X_10[i] - 1) * CAMC_GRID_SIZE \
                        + Lite_##PREFIX##_weight_Y_10[i] - 1; \
        SCORE += Lite_##PREFIX##_weight_10[i] * histogram[index]; \
    }

    ACCUMULATE(scores[0], 2PSK)
    ACCUMULATE(scores[1], 2PSK_45m)
    ACCUMULATE(scores[2], 2PSK_45p)
    ACCUMULATE(scores[3], 2PSK_90p)
    ACCUMULATE(scores[4], 4PSK)
    ACCUMULATE(scores[5], 4PSK_45m)
    ACCUMULATE(scores[6], 8PSK)
    ACCUMULATE(scores[7], 8PSK_45m)
    ACCUMULATE(scores[8], 16QAM)
    ACCUMULATE(scores[9], 16QAM_45m)

#undef ACCUMULATE

    result[0] = scores[0];
    for (int i = 1; i < 4; ++i) {
#pragma HLS PIPELINE
        if (scores[i] > result[0]) {
            result[0] = scores[i];
        }
    }
    result[1] = scores[4] > scores[5] ? scores[4] : scores[5];
    result[2] = scores[6] > scores[7] ? scores[6] : scores[7];
    result[3] = scores[8] > scores[9] ? scores[8] : scores[9];
}

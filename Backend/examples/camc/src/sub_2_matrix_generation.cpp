#include "lib.hpp"

void matrix_generation(
    const coordinate_t coordinates[CAMC_COORDINATE_SIZE],
    histogram_t histogram[CAMC_HISTOGRAM_SIZE]
) {
    for (int i = 0; i < CAMC_HISTOGRAM_SIZE; ++i) {
#pragma HLS PIPELINE
        histogram[i] = 0;
    }

    for (int i = 0; i < CAMC_SAMPLE_COUNT; ++i) {
#pragma HLS PIPELINE
        const coordinate_t x = coordinates[i];
        const coordinate_t y = coordinates[CAMC_SAMPLE_COUNT + i];
        if (x > 0 && y > 0 && x < CAMC_GRID_SIZE && y < CAMC_GRID_SIZE) {
            const int index = (x - 1) * CAMC_GRID_SIZE + y - 1;
            histogram[index] = histogram[index] + 1;
        }
    }
}

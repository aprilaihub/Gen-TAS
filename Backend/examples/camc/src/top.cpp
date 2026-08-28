#include "lib.hpp"

void camc_top(
    sample_t a[CAMC_INPUT_SIZE],
    score_t b[CAMC_SCORE_COUNT]
) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=a
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=b
#pragma HLS INTERFACE s_axilite port=a bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    coordinate_t coordinates[CAMC_COORDINATE_SIZE];
    histogram_t histogram[CAMC_HISTOGRAM_SIZE];

    axis_initialisation(a, coordinates);
    matrix_generation(coordinates, histogram);
    array_product(histogram, b);
}

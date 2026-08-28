#ifndef CAMC_LIB_HPP
#define CAMC_LIB_HPP

#include <ap_fixed.h>
#include <ap_int.h>

constexpr int CAMC_SAMPLE_COUNT = 800;
constexpr int CAMC_INPUT_SIZE = 2 * CAMC_SAMPLE_COUNT;
constexpr int CAMC_COORDINATE_SIZE = 2 * CAMC_SAMPLE_COUNT;
constexpr int CAMC_GRID_SIZE = 100;
constexpr int CAMC_HISTOGRAM_SIZE = CAMC_GRID_SIZE * CAMC_GRID_SIZE;
constexpr int CAMC_SCORE_COUNT = 4;

typedef ap_uint<20> sample_t;
typedef ap_uint<8> coordinate_t;
typedef ap_uint<14> histogram_t;
typedef ap_uint<64> score_t;

void axis_initialisation(
    const sample_t input[CAMC_INPUT_SIZE],
    coordinate_t coordinates[CAMC_COORDINATE_SIZE]
);

void matrix_generation(
    const coordinate_t coordinates[CAMC_COORDINATE_SIZE],
    histogram_t histogram[CAMC_HISTOGRAM_SIZE]
);

void array_product(
    const histogram_t histogram[CAMC_HISTOGRAM_SIZE],
    score_t result[CAMC_SCORE_COUNT]
);

void camc_top(
    sample_t a[CAMC_INPUT_SIZE],
    score_t b[CAMC_SCORE_COUNT]
);

#endif

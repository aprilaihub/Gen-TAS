#include "lib.hpp"

static coordinate_t quantize_axis(ap_int<20> input) {
    ap_fixed<20, 8> axis_final;
    ap_fixed<20, 8> axis_fraction;
    ap_int<8> axis_integer;

    const ap_uint<1> sign_bit = (input.range() >> 19) & 1;
    if (sign_bit == 1) {
        const ap_uint<20> magnitude = 1048576 - input;
        axis_integer = (magnitude.range() >> 12) & 255;
        axis_fraction = (magnitude.range() & 4095) * 0.000244140625;
        axis_final = 5 - (axis_fraction + axis_integer);
    } else {
        axis_integer = (input.range() >> 12) & 255;
        axis_fraction = (input.range() & 4095) * 0.000244140625;
        axis_final = axis_fraction + axis_integer + 5;
    }

    if (axis_final < 0 || axis_final >= 16) {
        return 0;
    }

    ap_ufixed<20, 8> scaled = axis_final * 10;
    coordinate_t rounded = scaled.range() >> 12;
    if (scaled - rounded >= 0.5 && scaled < CAMC_GRID_SIZE) {
        ++rounded;
    } else if (rounded >= CAMC_GRID_SIZE) {
        rounded = 0;
    }
    return rounded;
}

void axis_initialisation(
    const sample_t input[CAMC_INPUT_SIZE],
    coordinate_t coordinates[CAMC_COORDINATE_SIZE]
) {
    for (int i = 0; i < CAMC_INPUT_SIZE; ++i) {
#pragma HLS PIPELINE
        coordinates[i] = quantize_axis(ap_int<20>(input[i]));
    }
}

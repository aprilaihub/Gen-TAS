#include <iostream>
#include "lib.hpp"

enum TestCase {
    ORIGINAL_16QAM,
    DISTRIBUTED_WIDE,
    BPSK_LIKE,
    FOUR_CORNERS,
    DISTRIBUTED_NARROW,
    TEST_CASE_COUNT
};

static const char *TEST_NAMES[TEST_CASE_COUNT] = {
    "original_16qam",
    "distributed_minus4_to_plus4",
    "bpsk_like",
    "four_corners_plus_minus2_5",
    "distributed_minus3_5_to_plus3_5"
};

static const unsigned long long EXPECTED[TEST_CASE_COUNT][CAMC_SCORE_COUNT] = {
    {0, 0, 0, 91121},
    {1993, 1866, 1590, 2035},
    {1058800, 756400, 314000, 0},
    {0, 0, 0, 0},
    {2121, 2572, 2098, 2341}
};

static void make_input(TestCase test_case, sample_t input[CAMC_INPUT_SIZE]) {
    const ap_fixed<20, 8> original_x[9] = {
        -2.91027081, -1.043975, 0.843665, -1.005242, 3.0777832,
        -2.987052, -2.97210, 3.10400, 0.9596310
    };
    const ap_fixed<20, 8> original_y[9] = {
        2.9327373, 2.879154, 3.0109, 2.920826, -3.1413207,
        -1.02065, -0.9094878, 1.150434, 2.984448
    };

    for (int i = 0; i < CAMC_SAMPLE_COUNT; ++i) {
        ap_fixed<20, 8> x;
        ap_fixed<20, 8> y;
        switch (test_case) {
        case ORIGINAL_16QAM:
            x = original_x[i % 9];
            y = original_y[i % 9];
            break;
        case DISTRIBUTED_WIDE:
            x = ap_fixed<20, 8>(((i * 37) % 801) / 100.0 - 4.0);
            y = ap_fixed<20, 8>(((i * 53 + 17) % 801) / 100.0 - 4.0);
            break;
        case BPSK_LIKE:
            x = ap_fixed<20, 8>((i & 1) ? 1.0 : -1.0);
            y = ap_fixed<20, 8>(0.0);
            break;
        case FOUR_CORNERS:
            x = ap_fixed<20, 8>((i & 1) ? 2.5 : -2.5);
            y = ap_fixed<20, 8>((i & 2) ? 2.5 : -2.5);
            break;
        case DISTRIBUTED_NARROW:
            x = ap_fixed<20, 8>(((i * 29 + 11) % 701) / 100.0 - 3.5);
            y = ap_fixed<20, 8>(((i * 71 + 23) % 701) / 100.0 - 3.5);
            break;
        default:
            x = 0;
            y = 0;
        }
        input[i] = x.range();
        input[CAMC_SAMPLE_COUNT + i] = y.range();
    }
}

int main() {
    int total_mismatches = 0;
    for (int test_index = 0; test_index < TEST_CASE_COUNT; ++test_index) {
        sample_t input[CAMC_INPUT_SIZE];
        score_t result[CAMC_SCORE_COUNT] = {0};
        make_input(static_cast<TestCase>(test_index), input);
        camc_top(input, result);

        std::cout << TEST_NAMES[test_index] << ":";
        for (int output = 0; output < CAMC_SCORE_COUNT; ++output) {
            std::cout << " " << result[output];
            if (result[output] != EXPECTED[test_index][output]) {
                std::cerr << "\nMismatch in " << TEST_NAMES[test_index]
                          << " output " << output << ": expected "
                          << EXPECTED[test_index][output] << ", got "
                          << result[output] << std::endl;
                ++total_mismatches;
            }
        }
        std::cout << std::endl;
    }

    if (total_mismatches != 0) {
        std::cerr << "FAIL: " << total_mismatches << " CAMC output mismatches."
                  << std::endl;
        return 1;
    }
    std::cout << "PASS: all " << TEST_CASE_COUNT
              << " CAMC golden test cases matched." << std::endl;
    return 0;
}

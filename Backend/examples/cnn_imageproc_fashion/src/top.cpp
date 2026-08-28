#include "lib.hpp"
#include "weights.hpp"

void cnn_imageproc_v2(
    data_t a[IMG_SIZE],
    data_t b[NUM_CLASSES]
) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=a
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=b
#pragma HLS INTERFACE s_axilite port=a bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    data_t conv1_out[CONV1_SIZE];
    data_t pool1_out[POOL1_SIZE];
    data_t conv2_out[CONV2_SIZE];
    data_t pool2_out[POOL2_SIZE];

#pragma HLS ARRAY_PARTITION variable=conv1_weights complete
#pragma HLS ARRAY_PARTITION variable=conv1_bias complete
#pragma HLS ARRAY_PARTITION variable=conv2_bias complete
#pragma HLS ARRAY_PARTITION variable=dense_weights cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=dense_bias complete
#pragma HLS ARRAY_PARTITION variable=conv1_out cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=pool1_out cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=pool2_out cyclic factor=16

    conv1_feature_extract(a, conv1_weights, conv1_bias, conv1_out);
    relu_pool1(conv1_out, pool1_out);
    conv2_feature_extract(pool1_out, conv2_weights, conv2_bias, conv2_out);
    relu_pool2(conv2_out, pool2_out);
    dense_classifier(pool2_out, dense_weights, dense_bias, b);
}

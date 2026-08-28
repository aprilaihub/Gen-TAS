#ifndef FASHION_CNN_IMAGEPROC_LIB_HPP
#define FASHION_CNN_IMAGEPROC_LIB_HPP

#include <ap_fixed.h>

// Fashion-MNIST CNN dimensions matching the trained PyTorch model:
// input 1x28x28 -> conv1 16x28x28 -> pool1 16x14x14
// -> conv2 32x14x14 -> pool2 32x7x7 -> dense 10 classes.
#define IMG_H 28
#define IMG_W 28
#define IN_CH 1
#define K 3
#define PAD 1

#define CONV1_OUT_CH 16
#define CONV1_H 28
#define CONV1_W 28
#define POOL1_H 14
#define POOL1_W 14

#define CONV2_OUT_CH 32
#define CONV2_H 14
#define CONV2_W 14
#define POOL2_H 7
#define POOL2_W 7

#define NUM_CLASSES 10
#define IMG_SIZE (IMG_H * IMG_W)
#define CONV1_SIZE (CONV1_OUT_CH * CONV1_H * CONV1_W)
#define POOL1_SIZE (CONV1_OUT_CH * POOL1_H * POOL1_W)
#define CONV2_SIZE (CONV2_OUT_CH * CONV2_H * CONV2_W)
#define POOL2_SIZE (CONV2_OUT_CH * POOL2_H * POOL2_W)
#define DENSE_IN_SIZE POOL2_SIZE

// Fixed-point types for inference.
// Fashion-MNIST pixels after torchvision ToTensor() are in [0, 1].
// Saturate at the representable limits instead of wrapping overflowing
// Conv2 activations/logits into values with the opposite sign.
typedef ap_fixed<16,4,AP_TRN,AP_SAT> data_t; // activations / logits (Q4.12)
typedef ap_fixed<16,2> weight_t;   // trained weights from weights.hpp
typedef ap_fixed<40,16> acc_t;     // wider accumulator for conv and dense sums

void cnn_imageproc_v2(data_t a[IMG_SIZE], data_t b[NUM_CLASSES]);

void conv1_feature_extract(
    data_t a[IMG_SIZE],
    const weight_t weights[CONV1_OUT_CH * IN_CH * K * K],
    const weight_t bias[CONV1_OUT_CH],
    data_t conv1_out[CONV1_SIZE]
);

void relu_pool1(
    data_t conv1_out[CONV1_SIZE],
    data_t pool1_out[POOL1_SIZE]
);

void conv2_feature_extract(
    data_t pool1_out[POOL1_SIZE],
    const weight_t weights[CONV2_OUT_CH * CONV1_OUT_CH * K * K],
    const weight_t bias[CONV2_OUT_CH],
    data_t conv2_out[CONV2_SIZE]
);

void relu_pool2(
    data_t conv2_out[CONV2_SIZE],
    data_t pool2_out[POOL2_SIZE]
);

void dense_classifier(
    data_t pool2_out[POOL2_SIZE],
    const weight_t weights[NUM_CLASSES * DENSE_IN_SIZE],
    const weight_t bias[NUM_CLASSES],
    data_t b[NUM_CLASSES]
);

#endif

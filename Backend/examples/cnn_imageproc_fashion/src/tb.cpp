#include <iostream>
#include "lib.hpp"
#include "fashion_samples.hpp"

using namespace std;

static int argmax(data_t logits[NUM_CLASSES]) {
    int pred = 0;
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (logits[c] > logits[pred]) pred = c;
    }
    return pred;
}

int main() {
    static const int expected_predictions[NUM_TEST_SAMPLES] = {
        9, 2, 1, 1, 0, 1, 4, 6, 5, 7, 4, 5, 5, 3, 4, 1
    };
    data_t a[IMG_SIZE];
    data_t b[NUM_CLASSES];
    int correct = 0;
    int prediction_mismatches = 0;

    cout << "Fashion-MNIST CNNImageProc ap_fixed testbench using embedded samples from FashionMNIST_data.zip" << endl;

    for (int s = 0; s < NUM_TEST_SAMPLES; s++) {
        // Normalize in floating point before converting to ap_fixed<16,4>.
        // Casting 255.0 to data_t first would wrap because data_t's maximum is < 8.
        for (int i = 0; i < IMG_SIZE; i++) {
            a[i] = data_t(static_cast<double>(sample_images[s][i]) / 255.0);
        }

        cnn_imageproc_v2(a, b);
        int pred = argmax(b);
        int label = sample_labels[s];
        if (pred == label) correct++;
        if (pred != expected_predictions[s]) {
            prediction_mismatches++;
            cerr << "Prediction regression at sample " << s
                 << ": expected=" << expected_predictions[s]
                 << " actual=" << pred << endl;
        }

        cout << "Sample " << s
             << " expected=" << label
             << " predicted=" << pred
             << " logits=[";
        for (int c = 0; c < NUM_CLASSES; c++) {
            cout << b[c].to_double();
            if (c != NUM_CLASSES - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    cout << "Correct: " << correct << "/" << NUM_TEST_SAMPLES << endl;
    if (prediction_mismatches != 0) {
        cerr << "FAIL: " << prediction_mismatches
             << " predictions differ from the fixed-point golden reference." << endl;
        return 1;
    }
    cout << "PASS: all predictions matched the fixed-point golden reference." << endl;
    return 0;
}

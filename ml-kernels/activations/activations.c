/*
 * Activation Functions for CARTS
 *
 * ReLU extracted from Darknet, GELU implemented from formula
 * Original Darknet: https://github.com/pjreddie/darknet
 *
 * Description:
 *   Activation functions introduce non-linearity in neural networks.
 *   This file implements the most common activation functions:
 *   - ReLU: max(0, x)
 *   - Leaky ReLU: max(0.1x, x)
 *   - GELU: Gaussian Error Linear Unit (used in transformers)
 *   - Softmax: Converts scores to probabilities (from llama2)
 */

#include <math.h>
#include <stdlib.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

// Problem size configuration
#ifndef SIZE
#define SIZE (1024 * 1024) // 1M elements
#endif

/*
 * ReLU Activation: f(x) = max(0, x)
 *
 * Most common activation function in CNNs.
 * Simple, fast, effective.
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_relu(float *x, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    x[i] = (x[i] > 0) ? x[i] : 0;
  }
}

/*
 * Leaky ReLU Activation: f(x) = max(αx, x) where α = 0.1
 *
 * Allows small negative values to pass through.
 * Helps with "dying ReLU" problem.
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_leaky(float *x, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    x[i] = (x[i] > 0) ? x[i] : 0.1f * x[i];
  }
}

/*
 * ReLU6 Activation: f(x) = min(max(0, x), 6)
 *
 * Used in mobile networks (MobileNet).
 * Limits maximum value to 6.
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_relu6(float *x, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    float val = x[i];
    val = (val > 0) ? val : 0;
    val = (val < 6) ? val : 6;
    x[i] = val;
  }
}

/*
 * GELU Activation: Gaussian Error Linear Unit
 *
 * f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * Used in transformers (BERT, GPT-2, GPT-3).
 * Smoother than ReLU, better gradient flow.
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_gelu(float *x, int n) {
  const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
  const float coeff = 0.044715f;

  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    float val = x[i];
    float cube = val * val * val;
    float inner = sqrt_2_over_pi * (val + coeff * cube);
    x[i] = 0.5f * val * (1.0f + tanhf(inner));
  }
}

/*
 * GELU Activation (Fast Approximation)
 *
 * f(x) = x * σ(1.702 * x)
 * where σ is sigmoid function
 *
 * Faster approximation of GELU.
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_gelu_fast(float *x, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    float val = x[i];
    // sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
    float sigmoid = 1.0f / (1.0f + expf(-1.702f * val));
    x[i] = val * sigmoid;
  }
}

/*
 * Sigmoid Activation: f(x) = 1 / (1 + exp(-x))
 *
 * Maps input to (0, 1) range.
 * Used in output layers for binary classification.
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_sigmoid(float *x, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    x[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

/*
 * Tanh Activation: f(x) = tanh(x)
 *
 * Maps input to (-1, 1) range.
 * Zero-centered (unlike sigmoid).
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void activate_tanh(float *x, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    x[i] = tanhf(x[i]);
  }
}

/*
 * Softmax Activation: Converts scores to probabilities
 *
 * Extracted from llama2-transformer.
 * Numerically stable implementation (subtracts max before exp).
 *
 * Parameters:
 *   x: Input/output array (modified in-place)
 *   n: Number of elements
 */
static void softmax(float *x, int n) {
  int i;

  // Find max value (for numerical stability)
  float max_val = x[0];
  for (i = 1; i < n; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  // Exp and sum
  float sum = 0.0f;
  for (i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // Normalize
  for (i = 0; i < n; i++) {
    x[i] /= sum;
  }
}

/*
 * Initialize test data with range covering negative and positive values
 */
static void init_data(float *x, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    // Range from -3 to 3
    x[i] = -3.0f + 6.0f * ((float)i / n);
  }
}

/*
 * Copy array
 */
static void copy_array(float *dst, float *src, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

int main(int argc, char **argv) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("activations");
  CARTS_STARTUP_TIMER_START("activations");

  int size = SIZE;
  int softmax_size = 100; // Smaller size for softmax (more interpretable)

  // Allocate memory
  float *input = (float *)malloc(size * sizeof(float));
  float *output = (float *)malloc(size * sizeof(float));
  float *softmax_input = (float *)malloc(softmax_size * sizeof(float));

  // Initialize data
  init_data(input, size);
  init_data(softmax_input, softmax_size);

  CARTS_STARTUP_TIMER_STOP();

  // Run all activations. Since output is reused, we compute each activation's
  // checksum immediately after it runs (before the next overwrites output).
  CARTS_KERNEL_TIMER_START("activations");

  double relu_checksum = 0.0;
  copy_array(output, input, size);
  activate_relu(output, size);
  for (int i = 0; i < size; i++) {
    relu_checksum += fabs(output[i]);
  }

  double leaky_checksum = 0.0;
  copy_array(output, input, size);
  activate_leaky(output, size);
  for (int i = 0; i < size; i++) {
    leaky_checksum += fabs(output[i]);
  }

  double relu6_checksum = 0.0;
  copy_array(output, input, size);
  activate_relu6(output, size);
  for (int i = 0; i < size; i++) {
    relu6_checksum += fabs(output[i]);
  }

  double gelu_checksum = 0.0;
  copy_array(output, input, size);
  activate_gelu(output, size);
  for (int i = 0; i < size; i++) {
    gelu_checksum += fabs(output[i]);
  }

  double gelu_fast_checksum = 0.0;
  copy_array(output, input, size);
  activate_gelu_fast(output, size);
  for (int i = 0; i < size; i++) {
    gelu_fast_checksum += fabs(output[i]);
  }

  double sigmoid_checksum = 0.0;
  copy_array(output, input, size);
  activate_sigmoid(output, size);
  for (int i = 0; i < size; i++) {
    sigmoid_checksum += fabs(output[i]);
  }

  double tanh_checksum = 0.0;
  copy_array(output, input, size);
  activate_tanh(output, size);
  for (int i = 0; i < size; i++) {
    tanh_checksum += fabs(output[i]);
  }

  double softmax_checksum = 0.0;
  softmax(softmax_input, softmax_size);
  for (int i = 0; i < softmax_size; i++) {
    softmax_checksum += fabs(softmax_input[i]);
  }

  CARTS_KERNEL_TIMER_STOP("activations");

  CARTS_VERIFICATION_TIMER_START("activations");
  double final_checksum = relu_checksum + leaky_checksum + relu6_checksum +
                          gelu_checksum + gelu_fast_checksum + sigmoid_checksum +
                          tanh_checksum + softmax_checksum;
  CARTS_BENCH_CHECKSUM(final_checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  // Cleanup
  CARTS_CLEANUP_TIMER_START("activations");
  free(input);
  free(output);
  free(softmax_input);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

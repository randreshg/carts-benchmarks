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
#include "arts/utils/benchmarks/CartsBenchmarks.h"

// Problem size configuration
#ifndef SIZE
#define SIZE (1024 * 1024) // 1M elements
#endif

#ifndef NREPS
#define NREPS 1
#endif

/*
 * ReLU Activation: f(x) = max(0, x)
 */
static void activate_relu(const float *input, float *output, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    output[i] = (input[i] > 0) ? input[i] : 0;
  }
}

/*
 * Leaky ReLU Activation: f(x) = max(ax, x) where a = 0.1
 */
static void activate_leaky(const float *input, float *output, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    output[i] = (input[i] > 0) ? input[i] : 0.1f * input[i];
  }
}

/*
 * ReLU6 Activation: f(x) = min(max(0, x), 6)
 */
static void activate_relu6(const float *input, float *output, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    float val = input[i];
    val = (val > 0) ? val : 0;
    val = (val < 6) ? val : 6;
    output[i] = val;
  }
}

/*
 * GELU Activation: Gaussian Error Linear Unit
 *
 * f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
static void activate_gelu(const float *input, float *output, int n) {
  const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
  const float coeff = 0.044715f;

  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    float val = input[i];
    float cube = val * val * val;
    float inner = sqrt_2_over_pi * (val + coeff * cube);
    output[i] = 0.5f * val * (1.0f + tanhf(inner));
  }
}

/*
 * GELU Activation (Fast Approximation)
 *
 * f(x) = x * sigmoid(1.702 * x)
 */
static void activate_gelu_fast(const float *input, float *output, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    float val = input[i];
    float sigmoid = 1.0f / (1.0f + expf(-1.702f * val));
    output[i] = val * sigmoid;
  }
}

/*
 * Sigmoid Activation: f(x) = 1 / (1 + exp(-x))
 */
static void activate_sigmoid(const float *input, float *output, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    output[i] = 1.0f / (1.0f + expf(-input[i]));
  }
}

/*
 * Tanh Activation: f(x) = tanh(x)
 */
static void activate_tanh(const float *input, float *output, int n) {
  int i;
#pragma omp parallel for
  for (i = 0; i < n; ++i) {
    output[i] = tanhf(input[i]);
  }
}

/*
 * Softmax Activation: Converts scores to probabilities
 *
 * Numerically stable implementation (subtracts max before exp).
 */
static void softmax(const float *input, float *output, int n) {
  int i;

  // Find max value (for numerical stability)
  float max_val = input[0];
  for (i = 1; i < n; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  // Exp and sum
  float sum = 0.0f;
  for (i = 0; i < n; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  // Normalize
  for (i = 0; i < n; i++) {
    output[i] /= sum;
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

int main(int argc, char **argv) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("activations");
  CARTS_STARTUP_TIMER_START("activations");

  int size = SIZE;
  int softmax_size = 100; // Smaller size for softmax (more interpretable)

  // Allocate separate output arrays per activation for independent DBs.
  float *input = (float *)malloc(size * sizeof(float));
  float *out_relu = (float *)malloc(size * sizeof(float));
  float *out_leaky = (float *)malloc(size * sizeof(float));
  float *out_relu6 = (float *)malloc(size * sizeof(float));
  float *out_gelu = (float *)malloc(size * sizeof(float));
  float *out_gelu_fast = (float *)malloc(size * sizeof(float));
  float *out_sigmoid = (float *)malloc(size * sizeof(float));
  float *out_tanh = (float *)malloc(size * sizeof(float));
  float *softmax_input = (float *)malloc(softmax_size * sizeof(float));
  float *softmax_output = (float *)malloc(softmax_size * sizeof(float));

  // Initialize data
  init_data(input, size);
  init_data(softmax_input, softmax_size);

  CARTS_STARTUP_TIMER_STOP();

  // Run all activations under NREPS repetition loop for kernel timing.
  CARTS_KERNEL_TIMER_START("activations");
  for (int rep = 0; rep < NREPS; rep++) {
    activate_relu(input, out_relu, size);
    activate_leaky(input, out_leaky, size);
    activate_relu6(input, out_relu6, size);
    activate_gelu(input, out_gelu, size);
    activate_gelu_fast(input, out_gelu_fast, size);
    activate_sigmoid(input, out_sigmoid, size);
    activate_tanh(input, out_tanh, size);
    softmax(softmax_input, softmax_output, softmax_size);
    CARTS_KERNEL_TIMER_ACCUM("activations");
  }
  CARTS_KERNEL_TIMER_PRINT("activations");

  // Compute checksums (last rep results are still in output arrays).
  CARTS_VERIFICATION_TIMER_START("activations");

  double relu_checksum = 0.0;
  for (int i = 0; i < size; i++) relu_checksum += fabs(out_relu[i]);

  double leaky_checksum = 0.0;
  for (int i = 0; i < size; i++) leaky_checksum += fabs(out_leaky[i]);

  double relu6_checksum = 0.0;
  for (int i = 0; i < size; i++) relu6_checksum += fabs(out_relu6[i]);

  double gelu_checksum = 0.0;
  for (int i = 0; i < size; i++) gelu_checksum += fabs(out_gelu[i]);

  double gelu_fast_checksum = 0.0;
  for (int i = 0; i < size; i++) gelu_fast_checksum += fabs(out_gelu_fast[i]);

  double sigmoid_checksum = 0.0;
  for (int i = 0; i < size; i++) sigmoid_checksum += fabs(out_sigmoid[i]);

  double tanh_checksum = 0.0;
  for (int i = 0; i < size; i++) tanh_checksum += fabs(out_tanh[i]);

  double softmax_checksum = 0.0;
  for (int i = 0; i < softmax_size; i++) softmax_checksum += fabs(softmax_output[i]);

  double final_checksum = relu_checksum + leaky_checksum + relu6_checksum +
                          gelu_checksum + gelu_fast_checksum + sigmoid_checksum +
                          tanh_checksum + softmax_checksum;
  CARTS_BENCH_CHECKSUM(final_checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  // Cleanup
  CARTS_CLEANUP_TIMER_START("activations");
  free(input);
  free(out_relu);
  free(out_leaky);
  free(out_relu6);
  free(out_gelu);
  free(out_gelu_fast);
  free(out_sigmoid);
  free(out_tanh);
  free(softmax_input);
  free(softmax_output);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

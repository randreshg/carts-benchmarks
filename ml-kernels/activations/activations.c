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
#include <stdio.h>
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

/*
 * Print statistics
 */
static void print_stats(const char *name, float *x, int n, int print_samples) {
  // Compute min, max, mean
  float min_val = x[0];
  float max_val = x[0];
  double sum = 0.0;

  int i;
  for (i = 0; i < n; ++i) {
    if (x[i] < min_val)
      min_val = x[i];
    if (x[i] > max_val)
      max_val = x[i];
    sum += x[i];
  }

  float mean = sum / n;

  printf("\n%s:\n", name);
  printf("  Range: [%.6f, %.6f]\n", min_val, max_val);
  printf("  Mean: %.6f\n", mean);

  if (print_samples > 0) {
    printf("  First %d values: ", print_samples);
    for (i = 0; i < print_samples && i < n; ++i) {
      printf("%.4f ", x[i]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  CARTS_BENCHMARKS_START();

  int size = SIZE;
  int softmax_size = 100; // Smaller size for softmax (more interpretable)

  printf("Activation Functions\n");
  printf("====================\n");
  printf("Array size: %d elements\n", size);
  printf("Softmax size: %d elements\n", softmax_size);
  printf("\n");

  CARTS_E2E_TIMER_START("activations_setup");

  // Allocate memory
  float *input = (float *)malloc(size * sizeof(float));
  float *output = (float *)malloc(size * sizeof(float));
  float *softmax_input = (float *)malloc(softmax_size * sizeof(float));

  // Initialize data
  init_data(input, size);
  init_data(softmax_input, softmax_size);

  CARTS_E2E_TIMER_STOP();

  // Print input statistics
  print_stats("Input", input, size, 10);

  // Test ReLU
  printf("\n--- Testing ReLU ---\n");
  CARTS_E2E_TIMER_START("activations_relu");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("relu");
  activate_relu(output, size);
  CARTS_KERNEL_TIMER_STOP("relu");
  CARTS_E2E_TIMER_STOP();
  print_stats("ReLU Output", output, size, 10);

  // Compute checksum inline
  double relu_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    relu_checksum += output[i];
  }
  CARTS_BENCH_CHECKSUM(relu_checksum);

  // Test Leaky ReLU
  printf("\n--- Testing Leaky ReLU ---\n");
  CARTS_E2E_TIMER_START("activations_leaky_relu");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("leaky_relu");
  activate_leaky(output, size);
  CARTS_KERNEL_TIMER_STOP("leaky_relu");
  CARTS_E2E_TIMER_STOP();
  print_stats("Leaky ReLU Output", output, size, 10);

  // Compute checksum inline
  double leaky_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    leaky_checksum += output[i];
  }
  CARTS_BENCH_CHECKSUM(leaky_checksum);

  // Test ReLU6
  printf("\n--- Testing ReLU6 ---\n");
  CARTS_E2E_TIMER_START("activations_relu6");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("relu6");
  activate_relu6(output, size);
  CARTS_KERNEL_TIMER_STOP("relu6");
  CARTS_E2E_TIMER_STOP();
  print_stats("ReLU6 Output", output, size, 10);

  // Compute checksum inline
  double relu6_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    relu6_checksum += output[i];
  }
  CARTS_BENCH_CHECKSUM(relu6_checksum);

  // Test GELU
  printf("\n--- Testing GELU ---\n");
  CARTS_E2E_TIMER_START("activations_gelu");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("gelu");
  activate_gelu(output, size);
  CARTS_KERNEL_TIMER_STOP("gelu");
  CARTS_E2E_TIMER_STOP();
  print_stats("GELU Output", output, size, 10);

  // Compute checksum inline
  double gelu_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    gelu_checksum += output[i];
  }
  CARTS_BENCH_CHECKSUM(gelu_checksum);

  // Test GELU Fast
  printf("\n--- Testing GELU (Fast) ---\n");
  CARTS_E2E_TIMER_START("activations_gelu_fast");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("gelu_fast");
  activate_gelu_fast(output, size);
  CARTS_KERNEL_TIMER_STOP("gelu_fast");
  CARTS_E2E_TIMER_STOP();
  print_stats("GELU Fast Output", output, size, 10);

  // Compute checksum inline
  double gelu_fast_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    gelu_fast_checksum += output[i];
  }
  CARTS_BENCH_CHECKSUM(gelu_fast_checksum);

  // Test Sigmoid
  printf("\n--- Testing Sigmoid ---\n");
  CARTS_E2E_TIMER_START("activations_sigmoid");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("sigmoid");
  activate_sigmoid(output, size);
  CARTS_KERNEL_TIMER_STOP("sigmoid");
  CARTS_E2E_TIMER_STOP();
  print_stats("Sigmoid Output", output, size, 10);

  // Compute checksum inline
  double sigmoid_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    sigmoid_checksum += output[i];
  }
  CARTS_BENCH_CHECKSUM(sigmoid_checksum);

  // Test Tanh
  printf("\n--- Testing Tanh ---\n");
  CARTS_E2E_TIMER_START("activations_tanh");
  copy_array(output, input, size);
  CARTS_KERNEL_TIMER_START("tanh");
  activate_tanh(output, size);
  CARTS_KERNEL_TIMER_STOP("tanh");
  CARTS_E2E_TIMER_STOP();
  print_stats("Tanh Output", output, size, 10);

  // Compute checksum inline using sum of absolute values for stability.
  // Tanh output is zero-centered, so plain sum would be ~0.
  double tanh_checksum = 0.0;
  for (int i = 0; i < size; i++) {
    tanh_checksum += fabs(output[i]);
  }
  CARTS_BENCH_CHECKSUM(tanh_checksum);

  // Test Softmax
  printf("\n--- Testing Softmax ---\n");
  CARTS_E2E_TIMER_START("activations_softmax");
  CARTS_KERNEL_TIMER_START("softmax");
  softmax(softmax_input, softmax_size);
  CARTS_KERNEL_TIMER_STOP("softmax");
  CARTS_E2E_TIMER_STOP();
  print_stats("Softmax Output", softmax_input, softmax_size, 10);

  // Compute checksum inline
  double softmax_checksum = 0.0;
  for (int i = 0; i < softmax_size; i++) {
    softmax_checksum += softmax_input[i];
  }
  CARTS_BENCH_CHECKSUM(softmax_checksum);

  // Validate Softmax (should sum to 1)
  double softmax_sum = 0.0;
  for (int i = 0; i < softmax_size; ++i) {
    softmax_sum += softmax_input[i];
  }
  printf("  Softmax sum (should be 1.0): %.10f\n", softmax_sum);

  if (fabs(softmax_sum - 1.0) < 0.0001) {
    printf("  Softmax validation passed\n");
  } else {
    printf("  Softmax validation failed\n");
  }

  printf("\nAll activation functions completed successfully!\n");

  double final_checksum = relu_checksum + leaky_checksum + relu6_checksum +
                          gelu_checksum + gelu_fast_checksum + sigmoid_checksum +
                          tanh_checksum + softmax_checksum;
  CARTS_BENCH_CHECKSUM(final_checksum);

  // Cleanup
  free(input);
  free(output);
  free(softmax_input);

  return 0;
}

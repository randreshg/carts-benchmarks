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
 *
 * CARTS Compatibility:
 * - No global variables
 * - Clean parameter passing
 * - OpenMP parallelization
 * - Element-wise operations
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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
void activate_relu(float *x, int n) {
  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void activate_leaky(float *x, int n) {
  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void activate_relu6(float *x, int n) {
  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void activate_gelu(float *x, int n) {
  const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
  const float coeff = 0.044715f;

  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void activate_gelu_fast(float *x, int n) {
  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void activate_sigmoid(float *x, int n) {
  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void activate_tanh(float *x, int n) {
  int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
void softmax(float *x, int n) {
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
void init_data(float *x, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    // Range from -3 to 3
    x[i] = -3.0f + 6.0f * ((float)i / n);
  }
}

/*
 * Copy array
 */
void copy_array(float *dst, float *src, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

/*
 * Print statistics
 */
void print_stats(const char *name, float *x, int n, int print_samples) {
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
  int size = SIZE;
  int softmax_size = 100; // Smaller size for softmax (more interpretable)

  printf("Activation Functions\n");
  printf("====================\n");
  printf("Array size: %d elements\n", size);
  printf("Softmax size: %d elements\n", softmax_size);
  printf("\n");

  // Allocate memory
  float *input = (float *)malloc(size * sizeof(float));
  float *output = (float *)malloc(size * sizeof(float));
  float *softmax_input = (float *)malloc(softmax_size * sizeof(float));

  if (!input || !output || !softmax_input) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  // Initialize data
  init_data(input, size);
  init_data(softmax_input, softmax_size);

  // Print input statistics
  print_stats("Input", input, size, 10);

  // Test ReLU
  printf("\n--- Testing ReLU ---\n");
  copy_array(output, input, size);
  activate_relu(output, size);
  print_stats("ReLU Output", output, size, 10);

  // Test Leaky ReLU
  printf("\n--- Testing Leaky ReLU ---\n");
  copy_array(output, input, size);
  activate_leaky(output, size);
  print_stats("Leaky ReLU Output", output, size, 10);

  // Test ReLU6
  printf("\n--- Testing ReLU6 ---\n");
  copy_array(output, input, size);
  activate_relu6(output, size);
  print_stats("ReLU6 Output", output, size, 10);

  // Test GELU
  printf("\n--- Testing GELU ---\n");
  copy_array(output, input, size);
  activate_gelu(output, size);
  print_stats("GELU Output", output, size, 10);

  // Test GELU Fast
  printf("\n--- Testing GELU (Fast) ---\n");
  copy_array(output, input, size);
  activate_gelu_fast(output, size);
  print_stats("GELU Fast Output", output, size, 10);

  // Test Sigmoid
  printf("\n--- Testing Sigmoid ---\n");
  copy_array(output, input, size);
  activate_sigmoid(output, size);
  print_stats("Sigmoid Output", output, size, 10);

  // Test Tanh
  printf("\n--- Testing Tanh ---\n");
  copy_array(output, input, size);
  activate_tanh(output, size);
  print_stats("Tanh Output", output, size, 10);

  // Test Softmax
  printf("\n--- Testing Softmax ---\n");
  softmax(softmax_input, softmax_size);
  print_stats("Softmax Output", softmax_input, softmax_size, 10);

  // Validate Softmax (should sum to 1)
  double softmax_sum = 0.0;
  for (int i = 0; i < softmax_size; ++i) {
    softmax_sum += softmax_input[i];
  }
  printf("  Softmax sum (should be 1.0): %.10f\n", softmax_sum);

  if (fabsf(softmax_sum - 1.0) < 0.0001) {
    printf("  Softmax validation passed\n");
  } else {
    printf("  Softmax validation failed\n");
  }

  printf("\nAll activation functions completed successfully!\n");

  // Cleanup
  free(input);
  free(output);
  free(softmax_input);

  return 0;
}

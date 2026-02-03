/*
 * Batch Normalization Kernel for CARTS
 *
 * Extracted and adapted from Darknet framework
 * Original: https://github.com/pjreddie/darknet
 *
 * Description:
 *   Batch normalization normalizes activations across the batch dimension.
 *   For each channel, computes mean and variance across batch and spatial
 * dimensions, then normalizes, scales, and shifts the activations.
 *
 * Algorithm:
 *   For each channel c:
 *     1. Compute mean: μ_c = mean(x[b,c,h,w] for all b,h,w)
 *     2. Compute variance: σ²_c = var(x[b,c,h,w] for all b,h,w)
 *     3. Normalize: x̂[b,c,h,w] = (x[b,c,h,w] - μ_c) / sqrt(σ²_c + ε)
 *     4. Scale and shift: y[b,c,h,w] = γ_c * x̂[b,c,h,w] + β_c
 *
 * Memory Layout: NCHW (batch, channels, height, width)
 * Index calculation: index = b*filters*spatial + f*spatial + i
 * where spatial = height * width
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

// Problem size configuration
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif

#ifndef CHANNELS
#define CHANNELS 64
#endif

#ifndef HEIGHT
#define HEIGHT 32
#endif

#ifndef WIDTH
#define WIDTH 32
#endif

#define EPSILON 0.00001f

/*
 * Compute mean across batch and spatial dimensions for each channel
 *
 * Parameters:
 *   x: Input tensor [batch, filters, spatial]
 *   batch: Batch size
 *   filters: Number of channels
 *   spatial: Height * Width
 *   mean: Output mean for each channel [filters]
 */
static void mean_cpu(float ***x, int batch, int filters, int spatial, float *mean) {
  float scale = 1.0f / (batch * spatial);
  int i, j, k;

#pragma omp parallel for private(j, k)
  for (i = 0; i < filters; ++i) {
    mean[i] = 0;
    for (j = 0; j < batch; ++j) {
      for (k = 0; k < spatial; ++k) {
        mean[i] += x[j][i][k];
      }
    }
    mean[i] *= scale;
  }
}

/*
 * Compute variance across batch and spatial dimensions for each channel
 *
 * Parameters:
 *   x: Input tensor [batch, filters, spatial]
 *   mean: Mean for each channel [filters]
 *   batch: Batch size
 *   filters: Number of channels
 *   spatial: Height * Width
 *   variance: Output variance for each channel [filters]
 */
static void variance_cpu(float ***x, float *mean, int batch, int filters, int spatial,
                  float *variance) {
  float scale = 1.0f / (batch * spatial - 1);
  int i, j, k;

#pragma omp parallel for private(j, k)
  for (i = 0; i < filters; ++i) {
    variance[i] = 0;
    for (j = 0; j < batch; ++j) {
      for (k = 0; k < spatial; ++k) {
        float diff = x[j][i][k] - mean[i];
        variance[i] += diff * diff;
      }
    }
    variance[i] *= scale;
  }
}

/*
 * Normalize activations using mean and variance
 *
 * Parameters:
 *   x: Input/output tensor [batch, filters, spatial] (modified in-place)
 *   mean: Mean for each channel [filters]
 *   variance: Variance for each channel [filters]
 *   batch: Batch size
 *   filters: Number of channels
 *   spatial: Height * Width
 */
static void normalize_cpu(float ***x, float *mean, float *variance, int batch,
                   int filters, int spatial) {
  int b, f, i;

#pragma omp parallel for private(f, i)
  for (b = 0; b < batch; ++b) {
    for (f = 0; f < filters; ++f) {
      for (i = 0; i < spatial; ++i) {
        x[b][f][i] = (x[b][f][i] - mean[f]) / (sqrtf(variance[f] + EPSILON));
      }
    }
  }
}

/*
 * Scale normalized activations by learned scale parameters
 *
 * Parameters:
 *   output: Output tensor [batch, filters, spatial] (modified in-place)
 *   scales: Scale parameter for each channel [filters]
 *   batch: Batch size
 *   n: Number of channels
 *   size: Spatial size (height * width)
 */
static void scale_bias_cpu(float ***output, float *scales, int batch, int n, int size) {
  int i, j, b;

#pragma omp parallel for private(i, j)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[b][i][j] *= scales[i];
      }
    }
  }
}

/*
 * Add learned bias to scaled activations
 *
 * Parameters:
 *   output: Output tensor [batch, filters, spatial] (modified in-place)
 *   biases: Bias parameter for each channel [filters]
 *   batch: Batch size
 *   n: Number of channels
 *   size: Spatial size (height * width)
 */
static void add_bias_cpu(float ***output, float *biases, int batch, int n, int size) {
  int i, j, b;

#pragma omp parallel for private(i, j)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[b][i][j] += biases[i];
      }
    }
  }
}

/*
 * Complete batch normalization forward pass
 *
 * Parameters:
 *   x: Input tensor [batch, channels, spatial]
 *   output: Output tensor [batch, channels, spatial]
 *   scales: Scale parameters γ [channels]
 *   biases: Bias parameters β [channels]
 *   batch: Batch size
 *   channels: Number of channels
 *   spatial: Height * Width
 *   mean: Workspace for mean [channels]
 *   variance: Workspace for variance [channels]
 */
static void batchnorm_forward(float ***x, float ***output, float *scales, float *biases,
                       int batch, int channels, int spatial,
                       float *mean, float *variance) {
  int b, c, i;

  // Copy input to output (will be modified in-place)
#pragma omp parallel for private(c, i)
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channels; ++c) {
      for (i = 0; i < spatial; ++i) {
        output[b][c][i] = x[b][c][i];
      }
    }
  }

  // Compute statistics
  mean_cpu(output, batch, channels, spatial, mean);
  variance_cpu(output, mean, batch, channels, spatial, variance);

  // Normalize
  normalize_cpu(output, mean, variance, batch, channels, spatial);

  // Scale and shift
  scale_bias_cpu(output, scales, batch, channels, spatial);
  add_bias_cpu(output, biases, batch, channels, spatial);
}

/*
 * Initialize test data
 */
static void init_data(float ***x, float *scales, float *biases, int batch, int channels,
               int spatial) {
  int b, c, i;
  int total = batch * channels * spatial;
  int idx = 0;

  // Initialize input with some pattern
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channels; ++c) {
      for (i = 0; i < spatial; ++i) {
        x[b][c][i] = ((float)idx / total) * 2.0f - 1.0f; // Range [-1, 1]
        idx++;
      }
    }
  }

  // Initialize scales to 1.0 and biases to 0.0
  for (i = 0; i < channels; ++i) {
    scales[i] = 1.0f;
    biases[i] = 0.0f;
  }
}

int main(int argc, char **argv) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("batchnorm");

  int batch = BATCH_SIZE;
  int channels = CHANNELS;
  int height = HEIGHT;
  int width = WIDTH;
  int spatial = height * width;
  int total_size = batch * channels * spatial;

  printf("Batch Normalization Kernel\n");
  printf("===========================\n");
  printf("Configuration:\n");
  printf("  Batch size: %d\n", batch);
  printf("  Channels: %d\n", channels);
  printf("  Spatial: %d x %d = %d\n", height, width, spatial);
  printf("  Total elements: %d\n", total_size);
  printf("\n");

  // Allocate memory for 3D arrays
  float ***x = (float ***)malloc(batch * sizeof(float **));
  float ***output = (float ***)malloc(batch * sizeof(float **));
  float *scales = (float *)malloc(channels * sizeof(float));
  float *biases = (float *)malloc(channels * sizeof(float));
  float *mean = (float *)malloc(channels * sizeof(float));
  float *variance = (float *)malloc(channels * sizeof(float));

  for (int b = 0; b < batch; ++b) {
    x[b] = (float **)malloc(channels * sizeof(float *));
    output[b] = (float **)malloc(channels * sizeof(float *));
    for (int c = 0; c < channels; ++c) {
      x[b][c] = (float *)malloc(spatial * sizeof(float));
      output[b][c] = (float *)malloc(spatial * sizeof(float));
    }
  }

  // Initialize data
  init_data(x, scales, biases, batch, channels, spatial);

  // Run batch normalization
  printf("Running batch normalization...\n");
  // CARTS_KERNEL_TIMER_START("batchnorm");
  batchnorm_forward(x, output, scales, biases, batch, channels, spatial,
                    mean, variance);
  // CARTS_KERNEL_TIMER_STOP("batchnorm");

  // Validation: normalized output should have mean ≈ 0 and variance ≈ 1 (before
  // scale/bias)
  float *ref_mean = (float *)malloc(channels * sizeof(float));
  float *ref_variance = (float *)malloc(channels * sizeof(float));

  // Check a few statistics
  printf("\nValidation:\n");
  printf("  First 5 channel means: ");
  for (int i = 0; i < 5 && i < channels; ++i) {
    printf("%.6f ", mean[i]);
  }
  printf("\n");

  printf("  First 5 channel variances: ");
  for (int i = 0; i < 5 && i < channels; ++i) {
    printf("%.6f ", variance[i]);
  }
  printf("\n");

  printf("  First 5 output values: ");
  for (int i = 0; i < 5 && i < spatial; ++i) {
    printf("%.6f ", output[0][0][i]);
  }
  printf("\n");

  printf("\nBatch normalization completed successfully!\n");

  // Compute checksum inline using sum of absolute values for stability.
  // Normalized data is centered around 0, so plain sum would be ~0
  // and highly sensitive to floating-point rounding differences.
  double checksum = 0.0;
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      for (int s = 0; s < spatial; s++) {
        checksum += fabs((double)output[b][c][s]);
      }
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  // Cleanup
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      free(x[b][c]);
      free(output[b][c]);
    }
    free(x[b]);
    free(output[b]);
  }
  free(x);
  free(output);
  free(scales);
  free(biases);
  free(mean);
  free(variance);
  free(ref_mean);
  free(ref_variance);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

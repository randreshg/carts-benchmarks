/*
 * Pooling Kernels (Max and Average) for CARTS
 *
 * Extracted and adapted from Darknet framework
 * Original: https://github.com/pjreddie/darknet
 *
 * Description:
 *   Pooling operations downsample spatial dimensions by aggregating
 *   values within pooling windows. Max pooling takes the maximum value,
 *   while average pooling computes the mean.
 *
 * Algorithm (Max Pooling):
 *   For each output position (b, c, i, j):
 *     output[b,c,i,j] = max(input[b,c,i*stride+m,j*stride+n]
 *                           for m,n in [0, pool_size))
 *
 * Algorithm (Average Pooling):
 *   For each output position (b, c, i, j):
 *     output[b,c,i,j] = mean(input[b,c,i*stride+m,j*stride+n]
 *                            for m,n in [0, pool_size))
 *
 * Memory Layout: NCHW (batch, channels, height, width)
 * Typical configurations:
 *   - 2x2 pooling, stride 2: Halves spatial dimensions
 *   - 3x3 pooling, stride 2: Common in older architectures
 *
 * CARTS Compatibility:
 * - No global variables
 * - Clean parameter passing
 * - OpenMP parallelization
 * - Self-contained functions
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Problem size configuration
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif

#ifndef CHANNELS
#define CHANNELS 64
#endif

#ifndef HEIGHT
#define HEIGHT 64
#endif

#ifndef WIDTH
#define WIDTH 64
#endif

#ifndef POOL_SIZE
#define POOL_SIZE 2
#endif

#ifndef STRIDE
#define STRIDE 2
#endif

#ifndef PADDING
#define PADDING 0
#endif

/*
 * Max Pooling Forward Pass
 *
 * Parameters:
 *   input: Input tensor [batch, channels, in_spatial] where in_spatial = in_height * in_width
 *   output: Output tensor [batch, channels, out_spatial] where out_spatial = out_height * out_width
 *   batch: Batch size
 *   channels: Number of channels
 *   in_height: Input height
 *   in_width: Input width
 *   pool_size: Size of pooling window (pool_size x pool_size)
 *   stride: Stride for pooling window
 *   padding: Padding (typically 0 for pooling)
 */
void maxpool_forward(float ***input, float ***output, int batch, int channels,
                     int in_height, int in_width, int pool_size, int stride,
                     int padding) {
  // Calculate output dimensions
  int out_height = (in_height + padding - pool_size) / stride + 1;
  int out_width = (in_width + padding - pool_size) / stride + 1;
  int in_spatial = in_height * in_width;
  int out_spatial = out_height * out_width;

  int w_offset = -padding / 2;
  int h_offset = -padding / 2;

  int b, c, i, j, m, n;

#ifdef _OPENMP
#pragma omp parallel for private(c, i, j, m, n) collapse(2)
#endif
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channels; ++c) {
      for (i = 0; i < out_height; ++i) {
        for (j = 0; j < out_width; ++j) {
          // Output spatial index
          int out_spatial_idx = i * out_width + j;

          // Find maximum in pooling window
          float max_val = -FLT_MAX;

          for (n = 0; n < pool_size; ++n) {
            for (m = 0; m < pool_size; ++m) {
              int cur_h = h_offset + i * stride + n;
              int cur_w = w_offset + j * stride + m;

              // Check if within bounds
              int valid = (cur_h >= 0 && cur_h < in_height && cur_w >= 0 &&
                           cur_w < in_width);

              if (valid) {
                int in_spatial_idx = cur_h * in_width + cur_w;
                float val = input[b][c][in_spatial_idx];
                if (val > max_val) {
                  max_val = val;
                }
              }
            }
          }

          output[b][c][out_spatial_idx] = max_val;
        }
      }
    }
  }
}

/*
 * Average Pooling Forward Pass
 *
 * Parameters:
 *   input: Input tensor [batch, channels, in_spatial] where in_spatial = in_height * in_width
 *   output: Output tensor [batch, channels, out_spatial] where out_spatial = out_height * out_width
 *   batch: Batch size
 *   channels: Number of channels
 *   in_height: Input height
 *   in_width: Input width
 *   pool_size: Size of pooling window (pool_size x pool_size)
 *   stride: Stride for pooling window
 *   padding: Padding (typically 0 for pooling)
 */
void avgpool_forward(float ***input, float ***output, int batch, int channels,
                     int in_height, int in_width, int pool_size, int stride,
                     int padding) {
  // Calculate output dimensions
  int out_height = (in_height + padding - pool_size) / stride + 1;
  int out_width = (in_width + padding - pool_size) / stride + 1;
  int in_spatial = in_height * in_width;
  int out_spatial = out_height * out_width;

  int w_offset = -padding / 2;
  int h_offset = -padding / 2;

  int b, c, i, j, m, n;

#ifdef _OPENMP
#pragma omp parallel for private(c, i, j, m, n) collapse(2)
#endif
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channels; ++c) {
      for (i = 0; i < out_height; ++i) {
        for (j = 0; j < out_width; ++j) {
          // Output spatial index
          int out_spatial_idx = i * out_width + j;

          // Compute average in pooling window
          float sum = 0.0f;
          int count = 0;

          for (n = 0; n < pool_size; ++n) {
            for (m = 0; m < pool_size; ++m) {
              int cur_h = h_offset + i * stride + n;
              int cur_w = w_offset + j * stride + m;

              // Check if within bounds
              int valid = (cur_h >= 0 && cur_h < in_height && cur_w >= 0 &&
                           cur_w < in_width);

              if (valid) {
                int in_spatial_idx = cur_h * in_width + cur_w;
                sum += input[b][c][in_spatial_idx];
                count++;
              }
            }
          }

          output[b][c][out_spatial_idx] = (count > 0) ? (sum / count) : 0.0f;
        }
      }
    }
  }
}

/*
 * Global Average Pooling
 * Averages across entire spatial dimensions (common before classification
 * layer)
 *
 * Parameters:
 *   input: Input tensor [batch, channels, spatial] where spatial = height * width
 *   output: Output tensor [batch, channels] (spatial dims averaged out)
 *   batch: Batch size
 *   channels: Number of channels
 *   height: Input height
 *   width: Input width
 */
void global_avgpool(float ***input, float **output, int batch, int channels,
                    int height, int width) {
  int spatial_size = height * width;
  int b, c, h, w;

#ifdef _OPENMP
#pragma omp parallel for private(c, h, w) collapse(2)
#endif
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channels; ++c) {
      float sum = 0.0f;

      for (h = 0; h < height; ++h) {
        for (w = 0; w < width; ++w) {
          int spatial_idx = h * width + w;
          sum += input[b][c][spatial_idx];
        }
      }

      output[b][c] = sum / spatial_size;
    }
  }
}

/*
 * Initialize test data
 */
void init_pooling_data(float ***input, int batch, int channels, int spatial) {
  int idx = 0;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial; ++s) {
        // Create some pattern that will show pooling effects
        input[b][c][s] = (float)(idx % 100) / 10.0f;
        idx++;
      }
    }
  }
}

/*
 * Print sample output for validation
 */
void print_sample(const char *name, float ***data, int batch, int channels,
                  int height, int width, int max_print) {
  printf("\n%s (first %d elements):\n", name, max_print);
  int count = 0;
  int spatial = height * width;
  for (int b = 0; b < batch && count < max_print; ++b) {
    for (int c = 0; c < channels && count < max_print; ++c) {
      for (int h = 0; h < height && count < max_print; ++h) {
        for (int w = 0; w < width && count < max_print; ++w) {
          int spatial_idx = h * width + w;
          printf("%.2f ", data[b][c][spatial_idx]);
          count++;
        }
      }
    }
  }
  printf("\n");
}

int main(int argc, char **argv) {
  int batch = BATCH_SIZE;
  int channels = CHANNELS;
  int in_height = HEIGHT;
  int in_width = WIDTH;
  int pool_size = POOL_SIZE;
  int stride = STRIDE;
  int padding = PADDING;

  int out_height = (in_height + padding - pool_size) / stride + 1;
  int out_width = (in_width + padding - pool_size) / stride + 1;

  int in_spatial = in_height * in_width;
  int out_spatial = out_height * out_width;

  printf("Pooling Kernels (Max and Average)\n");
  printf("==================================\n");
  printf("Configuration:\n");
  printf("  Batch size: %d\n", batch);
  printf("  Channels: %d\n", channels);
  printf("  Input spatial: %d x %d\n", in_height, in_width);
  printf("  Pool size: %d x %d\n", pool_size, pool_size);
  printf("  Stride: %d\n", stride);
  printf("  Padding: %d\n", padding);
  printf("  Output spatial: %d x %d\n", out_height, out_width);
  printf("  Reduction factor: %.2fx\n",
         (float)(in_height * in_width) / (out_height * out_width));
  printf("\n");

  // Allocate memory as 3D arrays
  float ***input = (float ***)malloc(batch * sizeof(float **));
  float ***maxpool_output = (float ***)malloc(batch * sizeof(float **));
  float ***avgpool_output = (float ***)malloc(batch * sizeof(float **));
  float **global_output = (float **)malloc(batch * sizeof(float *));

  if (!input || !maxpool_output || !avgpool_output || !global_output) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  for (int b = 0; b < batch; ++b) {
    input[b] = (float **)malloc(channels * sizeof(float *));
    maxpool_output[b] = (float **)malloc(channels * sizeof(float *));
    avgpool_output[b] = (float **)malloc(channels * sizeof(float *));
    global_output[b] = (float *)malloc(channels * sizeof(float));
    if (!input[b] || !maxpool_output[b] || !avgpool_output[b] || !global_output[b]) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
    for (int c = 0; c < channels; ++c) {
      input[b][c] = (float *)malloc(in_spatial * sizeof(float));
      maxpool_output[b][c] = (float *)malloc(out_spatial * sizeof(float));
      avgpool_output[b][c] = (float *)malloc(out_spatial * sizeof(float));
      if (!input[b][c] || !maxpool_output[b][c] || !avgpool_output[b][c]) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
      }
    }
  }

  // Initialize data
  init_pooling_data(input, batch, channels, in_spatial);

  // Run max pooling
  printf("Running max pooling...\n");
  maxpool_forward(input, maxpool_output, batch, channels, in_height, in_width,
                  pool_size, stride, padding);

  // Run average pooling
  printf("Running average pooling...\n");
  avgpool_forward(input, avgpool_output, batch, channels, in_height, in_width,
                  pool_size, stride, padding);

  // Run global average pooling
  printf("Running global average pooling...\n");
  global_avgpool(input, global_output, batch, channels, in_height, in_width);

  // Print samples
  print_sample("Input", input, batch, channels, in_height, in_width, 20);
  print_sample("Max Pool Output", maxpool_output, batch, channels, out_height,
               out_width, 20);
  print_sample("Avg Pool Output", avgpool_output, batch, channels, out_height,
               out_width, 20);

  printf(
      "\nGlobal Average Pooling Output (first 10 channels of first batch):\n");
  for (int i = 0; i < 10 && i < channels; ++i) {
    printf("%.4f ", global_output[0][i]);
  }
  printf("\n");

  // Validation: Check that max pool >= avg pool (element-wise)
  printf("\nValidation:\n");
  int violations = 0;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < out_spatial; ++s) {
        if (maxpool_output[b][c][s] <
            avgpool_output[b][c][s] - 0.001f) { // Small tolerance for floating point
          violations++;
        }
      }
    }
  }

  if (violations == 0) {
    printf("  ✓ Max pool >= Avg pool for all elements (as expected)\n");
  } else {
    printf("  ✗ Found %d violations where max < avg\n", violations);
  }

  printf("\nPooling operations completed successfully!\n");

  // Cleanup
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      free(input[b][c]);
      free(maxpool_output[b][c]);
      free(avgpool_output[b][c]);
    }
    free(input[b]);
    free(maxpool_output[b]);
    free(avgpool_output[b]);
    free(global_output[b]);
  }
  free(input);
  free(maxpool_output);
  free(avgpool_output);
  free(global_output);

  return 0;
}

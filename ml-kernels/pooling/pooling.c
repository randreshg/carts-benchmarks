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
 */

#include <float.h>
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
static void maxpool_forward(float ***input, float ***output, int batch, int channels,
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

#pragma omp parallel for private(c, i, j, m, n)
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
static void avgpool_forward(float ***input, float ***output, int batch, int channels,
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

#pragma omp parallel for private(c, i, j, m, n)
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
static void global_avgpool(float ***input, float **output, int batch, int channels,
                    int height, int width) {
  int spatial_size = height * width;
  int b, c, h, w;

#pragma omp parallel for private(c, h, w)
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
static void init_pooling_data(float ***input, int batch, int channels, int spatial) {
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

int main(int argc, char **argv) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("pooling");
  CARTS_STARTUP_TIMER_START("pooling");

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

  // Allocate memory as 3D arrays
  float ***input = (float ***)malloc(batch * sizeof(float **));
  float ***maxpool_output = (float ***)malloc(batch * sizeof(float **));
  float ***avgpool_output = (float ***)malloc(batch * sizeof(float **));
  float **global_output = (float **)malloc(batch * sizeof(float *));

  for (int b = 0; b < batch; ++b) {
    input[b] = (float **)malloc(channels * sizeof(float *));
    maxpool_output[b] = (float **)malloc(channels * sizeof(float *));
    avgpool_output[b] = (float **)malloc(channels * sizeof(float *));
    global_output[b] = (float *)malloc(channels * sizeof(float));
    for (int c = 0; c < channels; ++c) {
      input[b][c] = (float *)malloc(in_spatial * sizeof(float));
      maxpool_output[b][c] = (float *)malloc(out_spatial * sizeof(float));
      avgpool_output[b][c] = (float *)malloc(out_spatial * sizeof(float));
    }
  }

  // Initialize data
  init_pooling_data(input, batch, channels, in_spatial);

  CARTS_STARTUP_TIMER_STOP();

  // Run all pooling kernels
  CARTS_KERNEL_TIMER_START("pooling");
  maxpool_forward(input, maxpool_output, batch, channels, in_height, in_width,
                  pool_size, stride, padding);
  avgpool_forward(input, avgpool_output, batch, channels, in_height, in_width,
                  pool_size, stride, padding);
  global_avgpool(input, global_output, batch, channels, in_height, in_width);
  CARTS_KERNEL_TIMER_STOP("pooling");

  // Compute checksums
  CARTS_VERIFICATION_TIMER_START("pooling");
  double maxpool_checksum = 0.0;
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      for (int s = 0; s < out_spatial; s++) {
        maxpool_checksum += maxpool_output[b][c][s];
      }
    }
  }

  double avgpool_checksum = 0.0;
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      for (int s = 0; s < out_spatial; s++) {
        avgpool_checksum += avgpool_output[b][c][s];
      }
    }
  }

  double global_avgpool_checksum = 0.0;
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      global_avgpool_checksum += global_output[b][c];
    }
  }

  double final_checksum = maxpool_checksum + avgpool_checksum + global_avgpool_checksum;
  CARTS_BENCH_CHECKSUM(final_checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  // Cleanup
  CARTS_CLEANUP_TIMER_START("pooling");
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
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

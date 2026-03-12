#include "arts/utils/benchmarks/CartsBenchmarks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BATCH
#define BATCH 16
#endif

#ifndef HIDDEN
#define HIDDEN 1024
#endif

#ifndef EPS
#define EPS 1e-5f
#endif

#ifndef NREPS
#define NREPS 1
#endif

static void init(float **x, float *gamma, float *beta) {
  int idx = 0;
  for (int b = 0; b < BATCH; ++b) {
    for (int h = 0; h < HIDDEN; ++h) {
      x[b][h] = ((float)(idx % 113) - 50.0f) * 0.03125f;
      idx++;
    }
  }
  for (int h = 0; h < HIDDEN; ++h) {
    gamma[h] = 1.0f;
    beta[h] = 0.0f;
  }
}

static void layernorm_forward(float **x, const float *gamma, const float *beta,
                              int batch, int hidden, float eps) {
#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    float mean = 0.0f;
    float var = 0.0f;
    for (int h = 0; h < hidden; ++h) {
      mean += x[b][h];
    }
    mean /= hidden;
    for (int h = 0; h < hidden; ++h) {
      float diff = x[b][h] - mean;
      var += diff * diff;
    }
    var = var / hidden;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int h = 0; h < hidden; ++h) {
      float norm = (x[b][h] - mean) * inv_std;
      x[b][h] = norm * gamma[h] + beta[h];
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("layernorm");

  CARTS_STARTUP_TIMER_START("layernorm");

  float **x = (float **)malloc(BATCH * sizeof(float *));
  float *gamma = (float *)malloc(sizeof(float) * HIDDEN);
  float *beta = (float *)malloc(sizeof(float) * HIDDEN);

  if (!x || !gamma || !beta) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  for (int b = 0; b < BATCH; ++b) {
    x[b] = (float *)malloc(HIDDEN * sizeof(float));
  }

  init(x, gamma, beta);

  CARTS_STARTUP_TIMER_STOP();

  CARTS_KERNEL_TIMER_START("layernorm");
  for (int rep = 0; rep < NREPS; rep++) {
    layernorm_forward(x, gamma, beta, BATCH, HIDDEN, EPS);
    CARTS_KERNEL_TIMER_ACCUM("layernorm");
  }
  CARTS_KERNEL_TIMER_PRINT("layernorm");

  CARTS_VERIFICATION_TIMER_START("layernorm");
  double checksum_value = 0.0;
  int diag = BATCH < HIDDEN ? BATCH : HIDDEN;
  for (int i = 0; i < diag; i++) {
    checksum_value += fabs((double)x[i][i]);
  }
  CARTS_BENCH_CHECKSUM(checksum_value);
  CARTS_VERIFICATION_TIMER_STOP();

  CARTS_CLEANUP_TIMER_START("layernorm");
  for (int b = 0; b < BATCH; ++b) {
    free(x[b]);
  }
  free(x);
  free(gamma);
  free(beta);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

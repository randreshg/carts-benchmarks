#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

#ifndef N_ELEMENTS
#define N_ELEMENTS 64
#endif

#ifndef N_BASIS
#define N_BASIS 30
#endif

#ifndef N_QUAD
#define N_QUAD 36
#endif

static void init(float **dofs, float **gradMatrix, float **fluxMatrix) {
  for (int e = 0; e < N_ELEMENTS; ++e) {
    for (int b = 0; b < N_BASIS; ++b) {
      int64_t idx = (int64_t)e * N_BASIS + b;
      int64_t val = (idx * 7) % 23;
      dofs[e][b] = 0.01f * (float)val;
    }
  }
  for (int q = 0; q < N_QUAD; ++q) {
    for (int b = 0; b < N_BASIS; ++b) {
      gradMatrix[q][b] = 0.001f * (float)(((q * N_BASIS + b) * 13) % 29);
      fluxMatrix[q][b] = 0.002f * (float)(((q * N_BASIS + b) * 17) % 31);
    }
  }
}

static void seissol_volume_integral(float **fluxOut,
                                    const float **dofs,
                                    const float **gradMatrix,
                                    const float **fluxMatrix) {
#pragma omp parallel for schedule(static)
  for (int elem = 0; elem < N_ELEMENTS; ++elem) {
    float buffer[N_QUAD];
    for (int q = 0; q < N_QUAD; ++q) {
      float val = 0.0f;
      for (int b = 0; b < N_BASIS; ++b) {
        val += gradMatrix[q][b] * dofs[elem][b];
      }
      buffer[q] = val;
    }
    for (int b = 0; b < N_BASIS; ++b) {
      float acc = 0.0f;
      for (int q = 0; q < N_QUAD; ++q) {
        acc += fluxMatrix[q][b] * buffer[q];
      }
      fluxOut[elem][b] = acc;
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();

  CARTS_E2E_TIMER_START("seissol_volume_integral");

  // Allocate 2D arrays
  float **dofs = (float **)malloc(N_ELEMENTS * sizeof(float *));
  float **gradMatrix = (float **)malloc(N_QUAD * sizeof(float *));
  float **fluxMatrix = (float **)malloc(N_QUAD * sizeof(float *));
  float **fluxOut = (float **)malloc(N_ELEMENTS * sizeof(float *));

  if (!dofs || !gradMatrix || !fluxMatrix || !fluxOut) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  for (int e = 0; e < N_ELEMENTS; ++e) {
    dofs[e] = (float *)malloc(N_BASIS * sizeof(float));
    fluxOut[e] = (float *)malloc(N_BASIS * sizeof(float));
  }

  for (int q = 0; q < N_QUAD; ++q) {
    gradMatrix[q] = (float *)malloc(N_BASIS * sizeof(float));
    fluxMatrix[q] = (float *)malloc(N_BASIS * sizeof(float));
  }

  init(dofs, gradMatrix, fluxMatrix);

  // CARTS_KERNEL_TIMER_START("seissol_volume_integral");
  seissol_volume_integral(fluxOut, dofs, gradMatrix, fluxMatrix);
  // CARTS_KERNEL_TIMER_STOP("seissol_volume_integral");

  // Compute checksum
  float checksum = 0.0f;
  for (int e = 0; e < N_ELEMENTS; ++e) {
    for (int b = 0; b < N_BASIS; ++b) {
      checksum += fluxOut[e][b];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  for (int e = 0; e < N_ELEMENTS; ++e) {
    free(dofs[e]);
    free(fluxOut[e]);
  }
  free(dofs);
  free(fluxOut);

  for (int q = 0; q < N_QUAD; ++q) {
    free(gradMatrix[q]);
    free(fluxMatrix[q]);
  }
  free(gradMatrix);
  free(fluxMatrix);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

/* PolyBench-like GEMM (single-file, OpenMP) */

#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NI
#define NI 512
#endif
#ifndef NJ
#define NJ 512
#endif
#ifndef NK
#define NK 512
#endif

static void init(float **A, float **B, float **C) {
  for (int i = 0; i < NI; i++)
    for (int k = 0; k < NK; k++)
      A[i][k] = (float)((i + k) % 13) * 0.01f;
  for (int k = 0; k < NK; k++)
    for (int j = 0; j < NJ; j++)
      B[k][j] = (float)((k * 3 + j) % 17) * 0.02f;
  for (int i = 0; i < NI; i++)
    for (int j = 0; j < NJ; j++)
      C[i][j] = 0.0f;
}

static void gemm(float **C, const float **A, const float **B, float alpha,
                 float beta) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
      float sum = 0.0f;
      for (int k = 0; k < NK; k++)
        sum += A[i][k] * B[k][j];
      C[i][j] = alpha * sum + beta * C[i][j];
    }
  }
}

static float checksum(float **C) {
  float s = 0.0f;
  for (int i = 0; i < NI; i++)
    for (int j = 0; j < NJ; j++)
      s += C[i][j];
  return s;
}

int main(void) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();

  // E2E timing: includes DB creation (malloc/init) + kernel
  CARTS_E2E_TIMER_START("gemm");

  float **A = (float **)malloc(NI * sizeof(float *));
  float **B = (float **)malloc(NK * sizeof(float *));
  float **C = (float **)malloc(NI * sizeof(float *));

  for (int i = 0; i < NI; i++) {
    A[i] = (float *)malloc(NK * sizeof(float));
    C[i] = (float *)malloc(NJ * sizeof(float));
  }
  for (int k = 0; k < NK; k++) {
    B[k] = (float *)malloc(NJ * sizeof(float));
  }

  init(A, B, C);

  // Kernel timing (just the compute kernel)
  CARTS_KERNEL_TIMER_START("gemm");
  gemm(C, (const float **)A, (const float **)B, 1.0f, 0.0f);
  CARTS_KERNEL_TIMER_STOP("gemm");

  // E2E stops after kernel, before verification/memfree
  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  // Verification (not timed)
  double checksum = 0.0;
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NJ; j++) {
      checksum += C[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  for (int i = 0; i < NI; i++) {
    free(A[i]);
    free(C[i]);
  }
  for (int k = 0; k < NK; k++) {
    free(B[k]);
  }
  free(A);
  free(B);
  free(C);
  return 0;
}

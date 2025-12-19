/* PolyBench-like 2D Jacobi */

#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1024
#endif
#ifndef TSTEPS
#define TSTEPS 100
#endif

int main(void) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();

  // E2E timing: includes DB creation (malloc/init) + kernel
  CARTS_E2E_TIMER_START("jacobi2d");

  float **A = (float **)malloc(N * sizeof(float *));
  float **B = (float **)malloc(N * sizeof(float *));

  for (int i = 0; i < N; i++) {
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
  }

  // Initialize arrays inline
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (float)((i + j) % 256) * 0.001f;
      B[i][j] = 0.0f;
    }
  }

  CARTS_KERNEL_TIMER_START("jacobi2d");

  for (int t = 0; t < TSTEPS; t += 2) {
    // Step 1: Read A, Write B
#pragma omp parallel for schedule(static)
    for (int i = 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        B[i][j] = 0.2f * (A[i][j] + A[i - 1][j] + A[i + 1][j] + A[i][j - 1] +
                          A[i][j + 1]);
      }
    }

    // Step 2: Read B, Write A (only if we have another iteration)
    if (t + 1 < TSTEPS) {
#pragma omp parallel for schedule(static)
      for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
          A[i][j] = 0.2f * (B[i][j] + B[i - 1][j] + B[i + 1][j] + B[i][j - 1] +
                            B[i][j + 1]);
        }
      }
    }
  }

  CARTS_KERNEL_TIMER_STOP("jacobi2d");

  // E2E stops after kernel, before verification/memfree
  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  // Verification (not timed) - result is in A if TSTEPS is even, B if odd
  double checksum = 0.0;
  if (TSTEPS % 2 == 0) {
    // Even TSTEPS: last write was to A
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        checksum += A[i][j];
      }
    }
  } else {
    // Odd TSTEPS: last write was to B
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        checksum += B[i][j];
      }
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  for (int i = 0; i < N; i++) {
    free(A[i]);
    free(B[i]);
  }
  free(A);
  free(B);
  return 0;
}

/* POLYBENCH - Seidel 2D
 */
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "seidel-2d.h"

#ifndef N
#define N 1000
#endif
#ifndef TSTEPS
#define TSTEPS 20
#endif
#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("seidel-2d");

  int n = N;
  int tsteps = TSTEPS;

  /* Pointer-to-pointer allocation */
  DATA_TYPE **A = (DATA_TYPE **)malloc(n * sizeof(DATA_TYPE *));
  for (int i = 0; i < n; i++) {
    A[i] = (DATA_TYPE *)malloc(n * sizeof(DATA_TYPE));
  }

  /* Initialize array inline */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
    }
  }

  // CARTS_KERNEL_TIMER_START("seidel-2d");

  /* Seidel-2D kernel - j loop is sequential due to A[i][j-1] dependency */
  for (int t = 0; t < tsteps; t++) {
#pragma omp parallel for schedule(static)
    for (int i = 1; i < n - 1; i++) {
      for (int j = 1; j < n - 1; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] +
                   A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] +
                   A[i + 1][j] + A[i + 1][j + 1]) /
                  9.0;
      }
    }
  }

  // CARTS_KERNEL_TIMER_STOP("seidel-2d");

  /* Verification */
  double checksum = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      checksum += A[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  /* Free arrays */
  for (int i = 0; i < n; i++) {
    free(A[i]);
  }
  free(A);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

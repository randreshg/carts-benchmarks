/* POLYBENCH/GPU-OPENMP - 3D Convolution
 * Rewritten to use pointer-to-pointer-to-pointer for CARTS compatibility
 */
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include "convolution-3d.h"
#include <omp.h>
#include <stdlib.h>

#ifndef NREPS
#define NREPS 1
#endif

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("convolution-3d");

  CARTS_STARTUP_TIMER_START("convolution-3d");
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Allocate 3D arrays using pointer-to-pointer-to-pointer */
  DATA_TYPE ***A = (DATA_TYPE ***)malloc(ni * sizeof(DATA_TYPE **));
  DATA_TYPE ***B = (DATA_TYPE ***)malloc(ni * sizeof(DATA_TYPE **));

  for (int i = 0; i < ni; i++) {
    A[i] = (DATA_TYPE **)malloc(nj * sizeof(DATA_TYPE *));
    B[i] = (DATA_TYPE **)malloc(nj * sizeof(DATA_TYPE *));
    for (int j = 0; j < nj; j++) {
      A[i][j] = (DATA_TYPE *)malloc(nk * sizeof(DATA_TYPE));
      B[i][j] = (DATA_TYPE *)malloc(nk * sizeof(DATA_TYPE));
    }
  }

  /* Initialize array A */
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      for (int k = 0; k < nk; k++) {
        A[i][j][k] = (DATA_TYPE)(i % 12 + 2 * (j % 7) + 3 * (k % 13));
        B[i][j][k] = 0.0f;
      }
    }
  }
  CARTS_STARTUP_TIMER_STOP();

  CARTS_KERNEL_TIMER_START("convolution-3d");
  for (int rep = 0; rep < NREPS; rep++) {

    /* 3D Convolution kernel */
#pragma omp parallel for schedule(static)
    for (int i = 1; i < ni - 1; ++i) {
      for (int j = 1; j < nj - 1; ++j) {
        for (int k = 1; k < nk - 1; ++k) {
          B[i][j][k] = 2 * A[i - 1][j - 1][k - 1] + 4 * A[i + 1][j - 1][k - 1] +
                       5 * A[i - 1][j - 1][k - 1] + 7 * A[i + 1][j - 1][k - 1] +
                       -8 * A[i - 1][j - 1][k - 1] + 10 * A[i + 1][j - 1][k - 1] +
                       -3 * A[i][j - 1][k] + 6 * A[i][j][k] +
                       -9 * A[i][j + 1][k] + 2 * A[i - 1][j - 1][k + 1] +
                       4 * A[i + 1][j - 1][k + 1] + 5 * A[i - 1][j][k + 1] +
                       7 * A[i + 1][j][k + 1] + -8 * A[i - 1][j + 1][k + 1] +
                       10 * A[i + 1][j + 1][k + 1];
        }
      }
    }

    CARTS_KERNEL_TIMER_ACCUM("convolution-3d");
  }
  CARTS_KERNEL_TIMER_PRINT("convolution-3d");

  /* Verification */
  CARTS_VERIFICATION_TIMER_START("convolution-3d");
  double checksum = 0.0;
  int diag = ni;
  if (nj < diag) diag = nj;
  if (nk < diag) diag = nk;
  for (int i = 0; i < diag; i++) {
    checksum += B[i][i][i];
  }
  CARTS_BENCH_CHECKSUM(checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  /* Free arrays */
  CARTS_CLEANUP_TIMER_START("convolution-3d");
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      free(A[i][j]);
      free(B[i][j]);
    }
    free(A[i]);
    free(B[i]);
  }
  free(A);
  free(B);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

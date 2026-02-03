/*
 * KaStORS Jacobi-For Benchmark
 * Adapted for CARTS compiler - Array-of-arrays version
 *
 * Original: https://github.com/viroulep/kastors
 * License: GNU LGPL
 * Author: John Burkardt (modified by KaStORS team)
 */

#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <stdio.h>
#include <stdlib.h>

static void sweep(int nx, int ny, double dx, double dy, double **f, int itold,
                  int itnew, double **u, double **unew, int block_size) {
  int i, j, it;

  for (it = itold + 1; it <= itnew; it++) {
    // Save the current estimate
#pragma omp parallel for private(j)
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        u[i][j] = unew[i][j];
      }
    }

    // Compute a new estimate
#pragma omp parallel for private(j)
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
          unew[i][j] = f[i][j];
        } else {
          unew[i][j] = 0.25 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] +
                               u[i + 1][j] + f[i][j] * dx * dy);
        }
      }
    }
  }
}

int main(void) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("jacobi-for");

#ifdef SIZE
  int nx = SIZE, ny = SIZE;
#else
  // default for testing
  int nx = 100, ny = 100;
#endif
  int itold = 0, itnew = 10;
  int block_size = 10;
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  // Allocate 2D arrays
  double **f = (double **)malloc(nx * sizeof(double *));
  double **u = (double **)malloc(nx * sizeof(double *));
  double **unew = (double **)malloc(nx * sizeof(double *));

  for (int i = 0; i < nx; i++) {
    f[i] = (double *)malloc(ny * sizeof(double));
    u[i] = (double *)malloc(ny * sizeof(double));
    unew[i] = (double *)malloc(ny * sizeof(double));
  }

  // Initialize arrays
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      f[i][j] = 0.0;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
    }
  }

  // CARTS_KERNEL_TIMER_START("jacobi-for");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);
  // CARTS_KERNEL_TIMER_STOP("jacobi-for");

  // Verification
  double checksum = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      checksum += unew[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  // Free 2D arrays
  for (int i = 0; i < nx; i++) {
    free(f[i]);
    free(u[i]);
    free(unew[i]);
  }
  free(f);
  free(u);
  free(unew);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

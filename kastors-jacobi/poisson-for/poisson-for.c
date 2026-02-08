/*
 * KaStORS Poisson-For Benchmark
 * Adapted for CARTS compiler - Array-of-arrays version
 *
 * Original: https://github.com/viroulep/kastors
 * License: GNU LGPL
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

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

static double u_exact(double x, double y) {
  double pi = 3.141592653589793;
  return sin(pi * x * y);
}

static double uxxyy_exact(double x, double y) {
  double pi = 3.141592653589793;
  return -pi * pi * (x * x + y * y) * sin(pi * x * y);
}

static void rhs(int nx, int ny, double **f, int block_size) {
  int i, j;
  double x, y;
  int nx1 = nx - 1;
  int ny1 = ny - 1;


#pragma omp parallel for private(i, j, x, y)
  for (j = 0; j < ny; j++) {
    y = (double)(j) / (double)ny1;
    for (i = 0; i < nx; i++) {
      x = (double)(i) / (double)nx1;
      if (i == 0 || i == nx1 || j == 0 || j == ny1)
        f[i][j] = u_exact(x, y);
      else
        f[i][j] = -uxxyy_exact(x, y);
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("poisson-for");

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

  // Set the right hand side array F
  rhs(nx, ny, f, block_size);

  // Set the initial solution estimate UNEW
  // We are "allowed" to pick up the boundary conditions exactly
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        unew[i][j] = f[i][j];
      } else {
        unew[i][j] = 0.0;
      }
    }
  }

  // CARTS_KERNEL_TIMER_START("sweep");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);
  // CARTS_KERNEL_TIMER_STOP("sweep");

  // Compute checksum
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

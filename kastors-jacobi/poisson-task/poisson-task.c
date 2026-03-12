/*
 * KaStORS Poisson-Task Benchmark
 * Adapted for CARTS compiler - Task-based version with dependencies
 *
 * Original: https://github.com/viroulep/kastors
 * License: GNU LGPL
 *
 * Solves: -DEL^2 U(x,y) = F(x,y) on unit square [0,1] x [0,1]
 * Exact solution: U(x,y) = sin(pi * x * y)
 */

#include <math.h>
#include <stdlib.h>
#include "arts/utils/benchmarks/CartsBenchmarks.h"

#ifndef SIZE
#define SIZE 100
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 10
#endif

#ifndef NITER
#define NITER 10
#endif

// Task-based sweep with row-level dependencies
static void sweep(int nx, int ny, double dx, double dy, double **f, int itold,
                  int itnew, double **u, double **unew) {
  int i, j, it;

#pragma omp parallel shared(u, unew, f) private(i, j, it)                      \
    firstprivate(nx, ny, dx, dy, itold, itnew)
#pragma omp single
  {
    for (it = itold + 1; it <= itnew; it++) {
      for (i = 0; i < nx; i++) {
#pragma omp task shared(u, unew) firstprivate(i) private(j)                    \
    depend(in : unew[i]) depend(out : u[i])
        for (j = 0; j < ny; j++) {
          u[i][j] = unew[i][j];
        }
      }
      for (i = 0; i < nx; i++) {
#pragma omp task shared(u, unew, f) firstprivate(i, nx, ny, dx, dy) private(j) \
    depend(in : f[i], u[i - 1], u[i], u[i + 1]) depend(out : unew[i])
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
}

// Initialize RHS: boundary = u_exact, interior = -uxxyy_exact
static void rhs(int nx, int ny, double **f) {
  double pi = 3.141592653589793;
  int i, j;
  double x, y;
  int nx1 = nx - 1;
  int ny1 = ny - 1;

#pragma omp parallel for private(i, x, y)
  for (j = 0; j < ny; j++) {
    y = (double)j / (double)ny1;
    for (i = 0; i < nx; i++) {
      x = (double)i / (double)nx1;
      if (i == 0 || i == nx1 || j == 0 || j == ny1) {
        // Boundary: u_exact(x,y) = sin(pi*x*y)
        f[i][j] = sin(pi * x * y);
      } else {
        // Interior: -uxxyy_exact = pi^2*(x^2+y^2)*sin(pi*x*y)
        f[i][j] = pi * pi * (x * x + y * y) * sin(pi * x * y);
      }
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("poisson-task");
  CARTS_STARTUP_TIMER_START("poisson-task");

  int nx = SIZE;
  int ny = SIZE;
  int itold = 0;
  int itnew = NITER;
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  // Allocate arrays
  double **f = (double **)malloc(nx * sizeof(double *));
  double **u = (double **)malloc(nx * sizeof(double *));
  double **unew = (double **)malloc(nx * sizeof(double *));

  for (int i = 0; i < nx; i++) {
    f[i] = (double *)malloc(ny * sizeof(double));
    u[i] = (double *)malloc(ny * sizeof(double));
    unew[i] = (double *)malloc(ny * sizeof(double));
  }

  // Initialize
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      f[i][j] = 0.0;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
    }
  }

  // Set RHS
  rhs(nx, ny, f);

  // Set initial estimate (boundary from f, interior = 0)
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        unew[i][j] = f[i][j];
      }
    }
  }

  CARTS_STARTUP_TIMER_STOP();

  CARTS_KERNEL_TIMER_START("poisson-task");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew);
  CARTS_KERNEL_TIMER_STOP("poisson-task");

  CARTS_VERIFICATION_TIMER_START("poisson-task");

  // Output checksum (diagonal sampling)
  double checksum = 0.0;
  int diag = nx < ny ? nx : ny;
  for (int i = 0; i < diag; i++) {
    checksum += fabs(unew[i][i]);
  }
  CARTS_BENCH_CHECKSUM(checksum);

  CARTS_VERIFICATION_TIMER_STOP();

  CARTS_CLEANUP_TIMER_START("poisson-task");

  // Cleanup
  for (int i = 0; i < nx; i++) {
    free(f[i]);
    free(u[i]);
    free(unew[i]);
  }
  free(f);
  free(u);
  free(unew);

  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

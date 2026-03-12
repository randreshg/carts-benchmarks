#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <math.h>
#include <stdlib.h>

// #pragma omp task depend version of SWEEP
static void sweep(int nx, int ny, double dx, double dy, double **f, int itold,
                  int itnew, double **u, double **unew, int block_size) {
  int i, it, j;

#pragma omp parallel shared(u, unew, f) private(i, j, it)                      \
    firstprivate(nx, ny, dx, dy, itold, itnew)
#pragma omp single
  {
    for (it = itold + 1; it <= itnew; it++) {
      // Save the current estimate.
      for (i = 0; i < nx; i++) {
#pragma omp task shared(u, unew) firstprivate(i) private(j)                    \
    depend(in : unew[i]) depend(out : u[i])
        for (j = 0; j < ny; j++) {
          u[i][j] = unew[i][j];
        }
      }
      // Compute a new estimate.
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

int main(void) {
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("jacobi-task-dep");
  CARTS_STARTUP_TIMER_START("jacobi-task-dep");

#ifdef SIZE
  int nx = SIZE;
  int ny = SIZE;
#else
  // default for testing
  int nx = 100;
  int ny = 100;
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
      f[i][j] = ((i + j) % 17 + 1) * 0.01;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
    }
  }

  CARTS_STARTUP_TIMER_STOP();

  CARTS_KERNEL_TIMER_START("jacobi-task-dep");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);
  CARTS_KERNEL_TIMER_STOP("jacobi-task-dep");

  CARTS_VERIFICATION_TIMER_START("jacobi-task-dep");

  // Compute checksum (diagonal sampling)
  double checksum = 0.0;
  int diag = nx < ny ? nx : ny;
  for (int i = 0; i < diag; i++) {
    checksum += unew[i][i];
  }
  CARTS_BENCH_CHECKSUM(checksum);

  CARTS_VERIFICATION_TIMER_STOP();

  CARTS_CLEANUP_TIMER_START("jacobi-task-dep");

  // Free 2D arrays
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

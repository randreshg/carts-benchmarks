#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Sequential version of sweep for verification
static void sweep_seq(int nx, int ny, double dx, double dy, double **f,
                      int itold, int itnew, double **u, double **unew) {
  for (int it = itold + 1; it <= itnew; it++) {
    // Save the current estimate
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        u[i][j] = unew[i][j];
      }
    }
    // Compute a new estimate
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
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
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();

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

  printf("Jacobi Task-Dep \n");
  printf("Grid size: %d x %d\n", nx, ny);
  printf("Iterations: %d\n", itnew);

  // E2E timing: includes DB creation (malloc/init) + kernel
  CARTS_E2E_TIMER_START("jacobi-task-dep");

  // Allocate 2D arrays
  double **f = (double **)malloc(nx * sizeof(double *));
  double **u = (double **)malloc(nx * sizeof(double *));
  double **unew = (double **)malloc(nx * sizeof(double *));
  double **unew_seq = (double **)malloc(nx * sizeof(double *));

  for (int i = 0; i < nx; i++) {
    f[i] = (double *)malloc(ny * sizeof(double));
    u[i] = (double *)malloc(ny * sizeof(double));
    unew[i] = (double *)malloc(ny * sizeof(double));
    unew_seq[i] = (double *)malloc(ny * sizeof(double));
  }

  // Initialize arrays
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      f[i][j] = 0.0;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
      unew_seq[i][j] = 0.0;
    }
  }

  printf("Running parallel version with task dependencies...\n");

  // Kernel timing (just the compute kernel)
  CARTS_KERNEL_TIMER_START("jacobi-task-dep");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);
  CARTS_KERNEL_TIMER_STOP("jacobi-task-dep");

  // E2E stops after kernel, before verification/memfree
  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  // Verification (not timed)
  // Save parallel result
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      unew_seq[i][j] = unew[i][j];
    }
  }

  // Re-initialize for sequential version
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
    }
  }

  printf("Running sequential version for verification...\n");
  sweep_seq(nx, ny, dx, dy, f, itold, itnew, u, unew);

  // Compare results
  double error = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      double diff = unew_seq[i][j] - unew[i][j];
      error += diff * diff;
    }
  }
  error = sqrt(error / (nx * ny));

  printf("Verification: %s (RMS error: %.2e)\n",
         (error < 1e-6) ? "PASS" : "FAIL", error);

  // Compute checksum
  double checksum = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      checksum += unew_seq[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  // Free 2D arrays
  for (int i = 0; i < nx; i++) {
    free(f[i]);
    free(u[i]);
    free(unew[i]);
    free(unew_seq[i]);
  }
  free(f);
  free(u);
  free(unew);
  free(unew_seq);

  return (error < 1e-6) ? 0 : 1;
}

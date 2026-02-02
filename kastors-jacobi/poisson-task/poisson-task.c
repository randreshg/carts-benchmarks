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
#include <stdio.h>
#include <stdlib.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

#ifdef POISSON_DEBUG
static double sum2d(int nx, int ny, double **a) {
  double s = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      s += a[i][j];
    }
  }
  return s;
}

static double boundary_sum(int nx, int ny, double **a) {
  double s = 0.0;
  for (int i = 0; i < nx; i++) {
    s += a[i][0];
    s += a[i][ny - 1];
  }
  for (int j = 1; j < ny - 1; j++) {
    s += a[0][j];
    s += a[nx - 1][j];
  }
  return s;
}

static double row_sum(int ny, double *row) {
  double s = 0.0;
  for (int j = 0; j < ny; j++) {
    s += row[j];
  }
  return s;
}

static void checkpoint(const char *label, int nx, int ny, double **a) {
  int mid = nx / 2;
  int last = nx - 1;
  printf("[ckpt] %s: sum=%.12e boundary=%.12e row0=%.12e rowMid=%.12e rowLast=%.12e\n",
         label, sum2d(nx, ny, a), boundary_sum(nx, ny, a), row_sum(ny, a[0]),
         row_sum(ny, a[mid]), row_sum(ny, a[last]));
  printf("[ckpt] %s samples: a[0][0]=%.6e a[0][1]=%.6e a[mid][mid]=%.6e a[last][last]=%.6e\n",
         label, a[0][0], a[0][1], a[mid][mid], a[last][last]);
}
#endif

#ifndef SIZE
#define SIZE 100
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 10
#endif

#ifndef NITER
#define NITER 10
#endif

// Sequential sweep for verification
static void sweep_seq(int nx, int ny, double dx, double dy, double **f,
                      int itold, int itnew, double **u, double **unew) {
  int i, j, it;
  for (it = itold + 1; it <= itnew; it++) {
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        u[i][j] = unew[i][j];
      }
    }
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

  int nx = SIZE;
  int ny = SIZE;
  int itold = 0;
  int itnew = NITER;
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  printf("Poisson Task\n");
  printf("Grid: %d x %d, Iterations: %d\n", nx, ny, itnew);
#ifdef POISSON_DEBUG
  printf("[ckpt] params: nx=%d ny=%d dx=%.12e dy=%.12e BLOCK_SIZE=%d NITER=%d\n",
         nx, ny, dx, dy, BLOCK_SIZE, NITER);
#endif

  CARTS_E2E_TIMER_START("poisson-task");

  // Allocate arrays
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

  // Initialize
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      f[i][j] = 0.0;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
      unew_seq[i][j] = 0.0;
    }
  }

  // Set RHS
  rhs(nx, ny, f);
#ifdef POISSON_DEBUG
  checkpoint("rhs", nx, ny, f);
#endif

  // Set initial estimate (boundary from f, interior = 0)
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        unew[i][j] = f[i][j];
      }
    }
  }
#ifdef POISSON_DEBUG
  checkpoint("init_unew", nx, ny, unew);
#endif

  printf("Running task-based sweep...\n");
  CARTS_KERNEL_TIMER_START("sweep");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew);
  CARTS_KERNEL_TIMER_STOP("sweep");

  CARTS_E2E_TIMER_STOP();
#ifdef POISSON_DEBUG
  checkpoint("task_unew", nx, ny, unew);
#endif

  // Save result
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      unew_seq[i][j] = unew[i][j];
    }
  }

  // Reset for sequential
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      u[i][j] = 0.0;
      unew[i][j] = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) ? f[i][j] : 0.0;
    }
  }

  printf("Running sequential sweep...\n");
  sweep_seq(nx, ny, dx, dy, f, itold, itnew, u, unew);
#ifdef POISSON_DEBUG
  checkpoint("seq_unew", nx, ny, unew);
#endif

  // Compare
  double error = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      double diff = unew_seq[i][j] - unew[i][j];
      error += diff * diff;
    }
  }
  error = sqrt(error / (nx * ny));

  printf("Verification: %s (RMS: %.2e)\n", (error < 1e-6) ? "PASS" : "FAIL", error);

  // Output checksum
  double checksum = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      checksum += fabs(unew_seq[i][j]);
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  // Cleanup
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

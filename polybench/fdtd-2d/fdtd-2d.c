/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 *
 * Copyright 2013, The University of Delaware
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 50x1000x1000. */
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include "fdtd-2d.h"

/* Array initialization. */
static void init_array(int tmax, int nx, int ny, DATA_TYPE **ex, DATA_TYPE **ey,
                       DATA_TYPE **hz, DATA_TYPE *_fict_) {
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE)i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      ex[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;
      ey[i][j] = ((DATA_TYPE)i * (j + 2)) / ny;
      hz[i][j] = ((DATA_TYPE)i * (j + 3)) / nx;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, int ny, DATA_TYPE **ex, DATA_TYPE **ey,
                        DATA_TYPE **hz) {
  int i, j;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, ex[i][j]);
      fprintf(stderr, DATA_PRINTF_MODIFIER, ey[i][j]);
      fprintf(stderr, DATA_PRINTF_MODIFIER, hz[i][j]);
      if ((i * nx + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_fdtd_2d(int tmax, int nx, int ny, DATA_TYPE **ex,
                           DATA_TYPE **ey, DATA_TYPE **hz, DATA_TYPE *_fict_) {
  int t, i, j;
#pragma scop
#pragma omp parallel private(t, i, j)
  {
    for (t = 0; t < _PB_TMAX; t++) {
#pragma omp for
      for (j = 0; j < _PB_NY; j++)
        ey[0][j] = _fict_[t];
#pragma omp for schedule(static)
      for (i = 1; i < _PB_NX; i++)
        for (j = 0; j < _PB_NY; j++)
          ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
#pragma omp for schedule(static)
      for (i = 0; i < _PB_NX; i++)
        for (j = 1; j < _PB_NY; j++)
          ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
#pragma omp for schedule(static)
      for (i = 0; i < _PB_NX - 1; i++)
        for (j = 0; j < _PB_NY - 1; j++)
          hz[i][j] = hz[i][j] -
                     0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
    }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("fdtd-2d");

  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  DATA_TYPE **ex = (DATA_TYPE **)malloc(nx * sizeof(DATA_TYPE *));
  DATA_TYPE **ey = (DATA_TYPE **)malloc(nx * sizeof(DATA_TYPE *));
  DATA_TYPE **hz = (DATA_TYPE **)malloc(nx * sizeof(DATA_TYPE *));
  DATA_TYPE *_fict_ = (DATA_TYPE *)malloc(tmax * sizeof(DATA_TYPE));

  for (int i = 0; i < nx; i++) {
    ex[i] = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
    ey[i] = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
    hz[i] = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(tmax, nx, ny, ex, ey, hz, _fict_);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  // CARTS_KERNEL_TIMER_START("fdtd-2d");
  kernel_fdtd_2d(tmax, nx, ny, ex, ey, hz, _fict_);
  // CARTS_KERNEL_TIMER_STOP("fdtd-2d");

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Verification */
  double checksum = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      checksum += ex[i][j];
      checksum += ey[i][j];
      checksum += hz[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, ex, ey, hz));

  /* Be clean. */
  for (int i = 0; i < nx; i++) {
    free(ex[i]);
    free(ey[i]);
    free(hz[i]);
  }
  free(ex);
  free(ey);
  free(hz);
  free(_fict_);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

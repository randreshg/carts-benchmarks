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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "arts/utils/benchmarks/CartsBenchmarks.h"
#include "atax.h"

#ifndef NREPS
#define NREPS 1
#endif

/* Array initialization. */
static void init_array(int nx, int ny, DATA_TYPE **A, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < ny; i++)
    x[i] = i * M_PI;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, DATA_TYPE *y) {
  int i;

  for (i = 0; i < nx; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_atax(int nx, int ny, DATA_TYPE **A, DATA_TYPE *x,
                        DATA_TYPE *y, DATA_TYPE *tmp) {
  int i, j;
#pragma scop
  /* Step 1: tmp = A * x */
#pragma omp parallel for private(j)
  for (i = 0; i < _PB_NX; i++) {
    tmp[i] = 0;
    for (j = 0; j < _PB_NY; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
  }

  /* Step 2: y = A^T * tmp */
#pragma omp parallel for private(i)
  for (j = 0; j < _PB_NY; j++) {
    y[j] = 0;
    for (i = 0; i < _PB_NX; i++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
#pragma endscop
}

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("atax");

  CARTS_STARTUP_TIMER_START("atax");
  /* Retrieve problem size. */
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  DATA_TYPE **A = (DATA_TYPE **)malloc(nx * sizeof(DATA_TYPE *));
  DATA_TYPE *x = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  DATA_TYPE *y = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  DATA_TYPE *tmp = (DATA_TYPE *)malloc(nx * sizeof(DATA_TYPE));

  for (int i = 0; i < nx; i++) {
    A[i] = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(nx, ny, A, x);
  CARTS_STARTUP_TIMER_STOP();

  /* Run kernel. */
  CARTS_KERNEL_TIMER_START("atax");
  for (int rep = 0; rep < NREPS; rep++) {
    kernel_atax(nx, ny, A, x, y, tmp);
    CARTS_KERNEL_TIMER_ACCUM("atax");
  }
  CARTS_KERNEL_TIMER_PRINT("atax");

  /* Verification */
  CARTS_VERIFICATION_TIMER_START("atax");
  double checksum = 0.0;
  for (int i = 0; i < ny; i++) {
    checksum += y[i];
  }
  CARTS_BENCH_CHECKSUM(checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  /* Be clean. */
  CARTS_CLEANUP_TIMER_START("atax");
  for (int i = 0; i < nx; i++) {
    free(A[i]);
  }
  free(A);
  free(x);
  free(y);
  free(tmp);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

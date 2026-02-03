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
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include "bicg.h"

/* Array initialization. */
static void init_array(int nx, int ny, DATA_TYPE **A, DATA_TYPE *r,
                       DATA_TYPE *p) {
  int i, j;

  for (i = 0; i < ny; i++)
    p[i] = i * M_PI;
  for (i = 0; i < nx; i++) {
    r[i] = i * M_PI;
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, int ny, DATA_TYPE *s, DATA_TYPE *q)

{
  int i;

  for (i = 0; i < ny; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, s[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  for (i = 0; i < nx; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, q[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_bicg(int nx, int ny, DATA_TYPE **A, DATA_TYPE *s,
                        DATA_TYPE *q, DATA_TYPE *p, DATA_TYPE *r) {
  int i, j;
#pragma scop
  /* Step 1: q = A * p */
#pragma omp parallel for private(j)
  for (i = 0; i < _PB_NX; i++) {
    q[i] = 0;
    for (j = 0; j < _PB_NY; j++)
      q[i] = q[i] + A[i][j] * p[j];
  }

  /* Step 2: s = A^T * r */
#pragma omp parallel for private(i)
  for (j = 0; j < _PB_NY; j++) {
    s[j] = 0;
    for (i = 0; i < _PB_NX; i++)
      s[j] = s[j] + r[i] * A[i][j];
  }
#pragma endscop
}

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("bicg");

  /* Retrieve problem size. */
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  DATA_TYPE **A = (DATA_TYPE **)malloc(nx * sizeof(DATA_TYPE *));
  DATA_TYPE *s = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  DATA_TYPE *q = (DATA_TYPE *)malloc(nx * sizeof(DATA_TYPE));
  DATA_TYPE *p = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  DATA_TYPE *r = (DATA_TYPE *)malloc(nx * sizeof(DATA_TYPE));

  if (!A || !s || !q || !p || !r) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  for (int i = 0; i < nx; i++) {
    A[i] = (DATA_TYPE *)malloc(ny * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(nx, ny, A, r, p);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  // CARTS_KERNEL_TIMER_START("bicg");
  kernel_bicg(nx, ny, A, s, q, p, r);
  // CARTS_KERNEL_TIMER_STOP("bicg");

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Verification */
  double checksum = 0.0;
  for (int i = 0; i < ny; i++) {
    checksum += s[i];
  }
  for (int i = 0; i < nx; i++) {
    checksum += q[i];
  }
  CARTS_BENCH_CHECKSUM(checksum);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, s, q));

  /* Be clean. */
  for (int i = 0; i < nx; i++) {
    free(A[i]);
  }
  free(A);
  free(s);
  free(q);
  free(p);
  free(r);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

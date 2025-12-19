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
/* Default data type is double, default size is 4096x4096. */
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include "convolution-2d.h"

/* Array initialization. */
static void init_array(int ni, int nj, DATA_TYPE **A) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      A[i][j] = ((DATA_TYPE)(i + j) / nj);
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, DATA_TYPE **B)

{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j]);
      if ((i * NJ + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_conv2d(int ni, int nj, DATA_TYPE **A, DATA_TYPE **B) {
  int i, j;
#pragma scop

#pragma omp parallel for private(j) schedule(static)
  for (i = 1; i < _PB_NI - 1; ++i) {
    for (j = 1; j < _PB_NJ - 1; ++j) {
      B[i][j] = 0.2 * A[i - 1][j - 1] + 0.5 * A[i - 1][j] +
                -0.8 * A[i - 1][j + 1] + -0.3 * A[i][j - 1] + 0.6 * A[i][j] +
                -0.9 * A[i][j + 1] + 0.4 * A[i + 1][j - 1] + 0.7 * A[i + 1][j] +
                0.1 * A[i + 1][j + 1];
    }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;

  // E2E timing: includes DB creation (malloc/init) + kernel
  CARTS_E2E_TIMER_START("convolution-2d");

  /* Variable declaration/allocation. */
  DATA_TYPE **A = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));
  DATA_TYPE **B = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));

  for (int i = 0; i < ni; i++) {
    A[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
    B[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(ni, nj, A);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  CARTS_KERNEL_TIMER_START("convolution-2d");
  kernel_conv2d(ni, nj, A, B);
  CARTS_KERNEL_TIMER_STOP("convolution-2d");

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  // E2E stops after kernel, before verification/memfree
  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  /* Verification (not timed) */
  double checksum = 0.0;
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      checksum += B[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, B));

  /* Be clean. */
  for (int i = 0; i < ni; i++) {
    free(A[i]);
    free(B[i]);
  }
  free(A);
  free(B);

  return 0;
}

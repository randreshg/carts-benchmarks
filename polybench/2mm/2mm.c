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
#include "2mm.h"
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

static void free_matrix(DATA_TYPE **matrix, int rows) {
  if (!matrix) {
    return;
  }
  for (int i = 0; i < rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha,
                       DATA_TYPE *beta, DATA_TYPE **A, DATA_TYPE **B,
                       DATA_TYPE **C, DATA_TYPE **D) {
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE)i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
  for (i = 0; i < nl; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = ((DATA_TYPE)i * (j + 3)) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = ((DATA_TYPE)i * (j + 2)) / nk;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl, DATA_TYPE **D) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, D[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_2mm(int ni, int nj, int nk, int nl, DATA_TYPE alpha,
                       DATA_TYPE beta, DATA_TYPE **tmp, DATA_TYPE **A,
                       DATA_TYPE **B, DATA_TYPE **C, DATA_TYPE **D) {
  int i, j, k;
#pragma scop
/* D := alpha*A*B*C + beta*D */
#pragma omp parallel
  {
#pragma omp for private(j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++) {
        tmp[i][j] = 0;
        for (k = 0; k < _PB_NK; ++k)
          tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
#pragma omp for private(j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NL; j++) {
        D[i][j] *= beta;
        for (k = 0; k < _PB_NJ; ++k)
          D[i][j] += tmp[i][k] * C[k][j];
      }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("2mm");

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE **tmp = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));
  DATA_TYPE **A = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));
  DATA_TYPE **B = (DATA_TYPE **)malloc(nk * sizeof(DATA_TYPE *));
  DATA_TYPE **C = (DATA_TYPE **)malloc(nl * sizeof(DATA_TYPE *));
  DATA_TYPE **D = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));

  for (int i = 0; i < ni; i++) {
    tmp[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
    A[i] = (DATA_TYPE *)malloc(nk * sizeof(DATA_TYPE));
    D[i] = (DATA_TYPE *)malloc(nl * sizeof(DATA_TYPE));
  }
  for (int i = 0; i < nk; i++) {
    B[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
  }
  for (int i = 0; i < nl; i++) {
    C[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  // CARTS_KERNEL_TIMER_START("2mm");
  kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
  // CARTS_KERNEL_TIMER_STOP("2mm");

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Verification */
  double checksum = 0.0;
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nl; j++) {
      checksum += D[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, D));

  /* Be clean. */
  free_matrix(tmp, ni);
  free_matrix(A, ni);
  free_matrix(B, nk);
  free_matrix(C, nl);
  free_matrix(D, ni);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

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
/* Default data type is double, default size is 4000. */
#include "3mm.h"
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE **A,
                       DATA_TYPE **B, DATA_TYPE **C, DATA_TYPE **D) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE)i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = ((DATA_TYPE)i * (j + 3)) / nl;
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = ((DATA_TYPE)i * (j + 2)) / nk;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl, DATA_TYPE **G) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, G[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_3mm(int ni, int nj, int nk, int nl, int nm, DATA_TYPE **E,
                       DATA_TYPE **A, DATA_TYPE **B, DATA_TYPE **F,
                       DATA_TYPE **C, DATA_TYPE **D, DATA_TYPE **G) {
  int i, j, k;
#pragma scop
#pragma omp parallel private(j, k)
  {
/* E := A*B */
#pragma omp for
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++) {
        E[i][j] = 0;
        for (k = 0; k < _PB_NK; ++k)
          E[i][j] += A[i][k] * B[k][j];
      }
/* F := C*D */
#pragma omp for
    for (i = 0; i < _PB_NJ; i++)
      for (j = 0; j < _PB_NL; j++) {
        F[i][j] = 0;
        for (k = 0; k < _PB_NM; ++k)
          F[i][j] += C[i][k] * D[k][j];
      }
/* G := E*F */
#pragma omp for
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NL; j++) {
        G[i][j] = 0;
        for (k = 0; k < _PB_NJ; ++k)
          G[i][j] += E[i][k] * F[k][j];
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
  int nk = NK;
  int nl = NL;
  int nm = NM;

  // E2E timing: includes DB creation (malloc/init) + kernel
  CARTS_E2E_TIMER_START("3mm");

  /* Variable declaration/allocation. */
  DATA_TYPE **E = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));
  DATA_TYPE **A = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));
  DATA_TYPE **B = (DATA_TYPE **)malloc(nk * sizeof(DATA_TYPE *));
  DATA_TYPE **F = (DATA_TYPE **)malloc(nj * sizeof(DATA_TYPE *));
  DATA_TYPE **C = (DATA_TYPE **)malloc(nj * sizeof(DATA_TYPE *));
  DATA_TYPE **D = (DATA_TYPE **)malloc(nm * sizeof(DATA_TYPE *));
  DATA_TYPE **G = (DATA_TYPE **)malloc(ni * sizeof(DATA_TYPE *));

  for (int i = 0; i < ni; i++) {
    E[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
    A[i] = (DATA_TYPE *)malloc(nk * sizeof(DATA_TYPE));
    G[i] = (DATA_TYPE *)malloc(nl * sizeof(DATA_TYPE));
  }
  for (int i = 0; i < nk; i++) {
    B[i] = (DATA_TYPE *)malloc(nj * sizeof(DATA_TYPE));
  }
  for (int i = 0; i < nj; i++) {
    F[i] = (DATA_TYPE *)malloc(nl * sizeof(DATA_TYPE));
    C[i] = (DATA_TYPE *)malloc(nm * sizeof(DATA_TYPE));
  }
  for (int i = 0; i < nm; i++) {
    D[i] = (DATA_TYPE *)malloc(nl * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, nm, A, B, C, D);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  CARTS_KERNEL_TIMER_START("3mm");
  kernel_3mm(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
  CARTS_KERNEL_TIMER_STOP("3mm");

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  // E2E stops after kernel, before verification/memfree
  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  /* Verification (not timed) */
  double checksum = 0.0;
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nl; j++) {
      checksum += G[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, G));

  /* Be clean. */
  for (int i = 0; i < ni; i++) {
    free(E[i]);
    free(A[i]);
    free(G[i]);
  }
  for (int i = 0; i < nk; i++) {
    free(B[i]);
  }
  for (int i = 0; i < nj; i++) {
    free(F[i]);
    free(C[i]);
  }
  for (int i = 0; i < nm; i++) {
    free(D[i]);
  }
  free(E);
  free(A);
  free(B);
  free(F);
  free(C);
  free(D);
  free(G);

  return 0;
}

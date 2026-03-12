/**
 * template.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "template-for-new-benchmark.h"
#include "arts/utils/benchmarks/CartsBenchmarks.h"


/* Array initialization. */
static
void init_array(int n, DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = 42;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_template(int n, DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      C[i][j] += 42;
#pragma endscop

}


int main(int argc, char** argv)
{
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("template");

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  CARTS_STARTUP_TIMER_START("template");
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(C));
  CARTS_STARTUP_TIMER_STOP();

  /* Run kernel. */
  CARTS_KERNEL_TIMER_START("template");
  kernel_template (n, POLYBENCH_ARRAY(C));
  CARTS_KERNEL_TIMER_STOP("template");

  /* Verification */
  CARTS_VERIFICATION_TIMER_START("template");
  /* TODO: compute checksum over live-out data (use diagonal sampling for O(n)) */
  double checksum = 0.0;
  CARTS_BENCH_CHECKSUM(checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  /* Be clean. */
  CARTS_CLEANUP_TIMER_START("template");
  POLYBENCH_FREE_ARRAY(C);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

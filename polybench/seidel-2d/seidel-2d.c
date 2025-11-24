/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 20x1000. */
#include "seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE **A)
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE **A)

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_seidel_2d(int tsteps,
		      int n,
		      DATA_TYPE **A)
{
  int t, i, j;

  #pragma scop
  #pragma omp parallel private (t,i,j)
  {
    for (t = 0; t <= _PB_TSTEPS - 1; t++) {
      #pragma omp for schedule(static) collapse (2)
      for (i = 1; i<= _PB_N - 2; i++) {
        for (j = 1; j <= _PB_N - 2; j++) {
          A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                     + A[i][j-1] + A[i][j] + A[i][j+1]
                     + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
        }
      }
    }
  }
  #pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  DATA_TYPE **A = (DATA_TYPE **)malloc(n * sizeof(DATA_TYPE *));
  
  if (!A) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }
  
  for (int i = 0; i < n; i++) {
    A[i] = (DATA_TYPE *)malloc(n * sizeof(DATA_TYPE));
    if (!A[i]) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
  }

  /* Initialize array(s). */
  init_array (n, A);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_seidel_2d (tsteps, n, A);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, A));

  /* Be clean. */
  for (int i = 0; i < n; i++) {
    free(A[i]);
  }
  free(A);

  return 0;
}

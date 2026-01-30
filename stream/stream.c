/*
 * STREAM benchmark adapted for CARTS benchmarks framework.
 * Based on STREAM version 5.10 by John D. McCalpin.
 *
 * Original: https://github.com/jeffhammond/STREAM
 * License: See original STREAM license.
 *
 * CARTS adaptation: No global variables - arrays passed as parameters.
 */

#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Array size - must be set at compile time via -DSTREAM_ARRAY_SIZE=N */
#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 10000000
#endif

/* Number of iterations for each kernel */
#ifndef NTIMES
#define NTIMES 10
#endif

/* Data type */
#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

/* Bytes per element */
#define BYTES_PER_WORD sizeof(STREAM_TYPE)

/* Number of kernels */
#define NUM_KERNELS 4

/* Initialize arrays with deterministic values */
static void init_arrays(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                        size_t array_size) {
#pragma omp parallel for schedule(static)
  for (size_t j = 0; j < array_size; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
}

/* Compute checksum for verification */
static double compute_checksum(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                               size_t array_size) {
  STREAM_TYPE asum = 0.0, bsum = 0.0, csum = 0.0;
  for (size_t j = 0; j < array_size; j++) {
    asum += a[j];
    bsum += b[j];
    csum += c[j];
  }
  return (double)(asum + bsum + csum);
}

int main(void) {
  size_t array_size = STREAM_ARRAY_SIZE;
  int ntimes = NTIMES;
  STREAM_TYPE scalar = 3.0;

  CARTS_BENCHMARKS_START();

  /* Print benchmark info */
  printf("-------------------------------------------------------------\n");
  printf("STREAM version adapted for CARTS benchmarks\n");
  printf("-------------------------------------------------------------\n");
  printf("Array size = %zu (elements)\n", array_size);
  printf("Memory per array = %.1f MiB (= %.1f GiB)\n",
         BYTES_PER_WORD * ((double)array_size / 1024.0 / 1024.0),
         BYTES_PER_WORD * ((double)array_size / 1024.0 / 1024.0 / 1024.0));
  printf("Total memory = %.1f MiB (= %.1f GiB)\n",
         (3.0 * BYTES_PER_WORD) * ((double)array_size / 1024.0 / 1024.0),
         (3.0 * BYTES_PER_WORD) * ((double)array_size / 1024.0 / 1024.0 / 1024.0));
  printf("Each kernel will be executed %d times.\n", ntimes);
  printf("-------------------------------------------------------------\n");

#ifdef _OPENMP
  printf("Number of Threads = %d\n", omp_get_max_threads());
#endif

  /* Allocate arrays - local variables, not global */
  STREAM_TYPE *a = (STREAM_TYPE *)malloc(array_size * sizeof(STREAM_TYPE));
  STREAM_TYPE *b = (STREAM_TYPE *)malloc(array_size * sizeof(STREAM_TYPE));
  STREAM_TYPE *c = (STREAM_TYPE *)malloc(array_size * sizeof(STREAM_TYPE));

  if (a == NULL || b == NULL || c == NULL) {
    printf("Error: Failed to allocate memory\n");
    return 1;
  }

  /* Timing storage - local variables */
  double times_copy[NTIMES], times_scale[NTIMES];
  double times_add[NTIMES], times_triad[NTIMES];

  /* Bytes moved per kernel iteration */
  double bytes_copy = 2.0 * BYTES_PER_WORD * array_size;
  double bytes_scale = 2.0 * BYTES_PER_WORD * array_size;
  double bytes_add = 3.0 * BYTES_PER_WORD * array_size;
  double bytes_triad = 3.0 * BYTES_PER_WORD * array_size;

  /* Initialize arrays */
  init_arrays(a, b, c, array_size);

  /* E2E timer includes all kernel iterations */
  CARTS_E2E_TIMER_START("stream");

  /*
   * Main timing loop - preserves original STREAM structure:
   * Each kernel runs NTIMES with individual timing per iteration.
   */
  for (int k = 0; k < ntimes; k++) {
    /* Copy: c[j] = a[j] */
    times_copy[k] = carts_bench_get_time();
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < array_size; j++)
      c[j] = a[j];
    times_copy[k] = carts_bench_get_time() - times_copy[k];

    /* Scale: b[j] = scalar * c[j] */
    times_scale[k] = carts_bench_get_time();
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < array_size; j++)
      b[j] = scalar * c[j];
    times_scale[k] = carts_bench_get_time() - times_scale[k];

    /* Add: c[j] = a[j] + b[j] */
    times_add[k] = carts_bench_get_time();
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < array_size; j++)
      c[j] = a[j] + b[j];
    times_add[k] = carts_bench_get_time() - times_add[k];

    /* Triad: a[j] = b[j] + scalar * c[j] */
    times_triad[k] = carts_bench_get_time();
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < array_size; j++)
      a[j] = b[j] + scalar * c[j];
    times_triad[k] = carts_bench_get_time() - times_triad[k];
  }

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  /* Calculate and print results (original STREAM format) */
  printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");

  /* Process each kernel's timing data */
  double *times_arr[NUM_KERNELS] = {times_copy, times_scale, times_add, times_triad};
  double bytes_arr[NUM_KERNELS] = {bytes_copy, bytes_scale, bytes_add, bytes_triad};
  const char *labels[NUM_KERNELS] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};
  const char *kernel_names[NUM_KERNELS] = {"copy", "scale", "add", "triad"};

  for (int j = 0; j < NUM_KERNELS; j++) {
    double avgtime = 0.0, maxtime = 0.0, mintime = DBL_MAX;
    for (int k = 1; k < ntimes; k++) { /* Skip first iteration */
      avgtime += times_arr[j][k];
      mintime = (times_arr[j][k] < mintime) ? times_arr[j][k] : mintime;
      maxtime = (times_arr[j][k] > maxtime) ? times_arr[j][k] : maxtime;
    }
    avgtime /= (double)(ntimes - 1);
    printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", labels[j],
           1.0E-06 * bytes_arr[j] / mintime, avgtime, mintime, maxtime);

    /* Also output CARTS-format kernel times */
    printf("kernel.%s: %.6fs\n", kernel_names[j], mintime);
  }
  printf("-------------------------------------------------------------\n");

  /* Verification */
  double checksum = compute_checksum(a, b, c, array_size);
  CARTS_BENCH_CHECKSUM(checksum);

  /* Cleanup */
  free(a);
  free(b);
  free(c);

  return 0;
}

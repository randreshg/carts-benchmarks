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

/* Compute checksum for verification (stride sampling) */
static double compute_checksum(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                               size_t array_size) {
  STREAM_TYPE asum = 0.0, bsum = 0.0, csum = 0.0;
  for (size_t j = 0; j < array_size; j += 128) {
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
  CARTS_E2E_TIMER_START("stream");

  CARTS_STARTUP_TIMER_START("stream");

  /* Allocate arrays - local variables, not global */
  STREAM_TYPE *a = (STREAM_TYPE *)malloc(array_size * sizeof(STREAM_TYPE));
  STREAM_TYPE *b = (STREAM_TYPE *)malloc(array_size * sizeof(STREAM_TYPE));
  STREAM_TYPE *c = (STREAM_TYPE *)malloc(array_size * sizeof(STREAM_TYPE));

  if (a == NULL || b == NULL || c == NULL) {
    return 1;
  }

  /* Timing storage - local variables */
  double times_copy[NTIMES], times_scale[NTIMES];
  double times_add[NTIMES], times_triad[NTIMES];

  /* Initialize arrays */
  init_arrays(a, b, c, array_size);

  CARTS_STARTUP_TIMER_STOP();

  CARTS_KERNEL_TIMER_START("stream");

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

  /* Print per-kernel timing (parsed by Python framework) */
  const char *kernel_names[NUM_KERNELS] = {"copy", "scale", "add", "triad"};
  double *times_arr[NUM_KERNELS] = {times_copy, times_scale, times_add, times_triad};
  for (int j = 0; j < NUM_KERNELS; j++) {
    double mintime = DBL_MAX;
    for (int k = 1; k < ntimes; k++) {
      mintime = (times_arr[j][k] < mintime) ? times_arr[j][k] : mintime;
    }
    printf("kernel.%s: %.6fs\n", kernel_names[j], mintime);
  }

  CARTS_KERNEL_TIMER_STOP("stream");

  CARTS_VERIFICATION_TIMER_START("stream");
  /* Verification */
  double checksum = compute_checksum(a, b, c, array_size);
  CARTS_BENCH_CHECKSUM(checksum);
  CARTS_VERIFICATION_TIMER_STOP();

  CARTS_CLEANUP_TIMER_START("stream");
  /* Cleanup */
  free(a);
  free(b);
  free(c);
  CARTS_CLEANUP_TIMER_STOP();

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

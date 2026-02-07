#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include "correlation.h"

static void print_array(int m, int n, DATA_TYPE **corr, const char *name) {
  (void)name;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, corr[i][j]);
      if ((i * n + j) % 20 == 0) {
        fprintf(stderr, "\n");
      }
    }
  }
  fprintf(stderr, "\n");
}

static void init_array(int m, int n, DATA_TYPE **data) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      data[i][j] = (DATA_TYPE)(i * j) / (DATA_TYPE)M + (DATA_TYPE)i / (j + 1);
    }
  }
}

static void kernel_correlation(int m, int n, DATA_TYPE **data, DATA_TYPE **corr,
                               DATA_TYPE *mean, DATA_TYPE *stddev) {
  /* Step 1: compute mean of each row. */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    mean[i] = 0.0;
    for (int j = 0; j < n; j++) {
      mean[i] += data[i][j];
    }
    mean[i] /= FLOAT_N;
  }

  /* Step 2: compute std deviation of each row. */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    stddev[i] = 0.0;
    for (int j = 0; j < n; j++) {
      DATA_TYPE diff = data[i][j] - mean[i];
      stddev[i] += diff * diff;
    }
    stddev[i] = sqrt(stddev[i] / FLOAT_N);
    if (stddev[i] <= EPS) {
      stddev[i] = 1.0;
    }
  }

  /* Step 3: center and scale the data. */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      data[i][j] = (data[i][j] - mean[i]) / (sqrt(FLOAT_N) * stddev[i]);
    }
  }

  /* Step 4: compute the correlation matrix. */
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    corr[i][i] = 1.0;
    for (int j = i + 1; j < m; j++) {
      DATA_TYPE sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += data[i][k] * data[j][k];
      }
      corr[i][j] = sum;
      corr[j][i] = sum;
    }
  }
}

int main(int argc, char **argv) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("correlation");

  int m = M;
  int n = N;

  DATA_TYPE **data = (DATA_TYPE **)malloc(m * sizeof(DATA_TYPE *));
  DATA_TYPE **corr = (DATA_TYPE **)malloc(m * sizeof(DATA_TYPE *));
  DATA_TYPE *mean = (DATA_TYPE *)malloc(m * sizeof(DATA_TYPE));
  DATA_TYPE *stddev = (DATA_TYPE *)malloc(m * sizeof(DATA_TYPE));

  for (int i = 0; i < m; i++) {
    data[i] = (DATA_TYPE *)malloc(n * sizeof(DATA_TYPE));
    corr[i] = (DATA_TYPE *)malloc(m * sizeof(DATA_TYPE));
  }

  init_array(m, n, data);

  // CARTS_KERNEL_TIMER_START("correlation");
  kernel_correlation(m, n, data, corr, mean, stddev);
  // CARTS_KERNEL_TIMER_STOP("correlation");

  /* Verification */
  double checksum = 0.0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      checksum += corr[i][j];
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  polybench_prevent_dce(print_array(m, m, corr, "corr"));

  for (int i = 0; i < m; i++) {
    free(data[i]);
    free(corr[i]);
  }
  free(data);
  free(corr);
  free(mean);
  free(stddev);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}

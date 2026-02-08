#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

#ifndef NX
#define NX 48
#endif
#ifndef NY
#define NY 48
#endif
#ifndef NZ
#define NZ 48
#endif
#ifndef POINTS_PER_DIR
#define POINTS_PER_DIR 4
#endif

#define COMP 3

static void init_array(float ****u, float ***mu, float ***lambda, float ****rhs) {
  int idx = 0;
  for (int c = 0; c < COMP; ++c) {
    idx = 0;
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
        for (int k = 0; k < NZ; ++k) {
          u[c][i][j][k] = 0.05f * (float)((idx + 17 * c) % 23) - 0.1f * (float)c;
          rhs[c][i][j][k] = 0.0f;
          idx++;
        }
      }
    }
  }
  idx = 0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        mu[i][j][k] = 3.0f + 0.001f * (float)(idx % 11);
        lambda[i][j][k] = 2.0f + 0.0015f * (float)(idx % 7);
        idx++;
      }
    }
  }
}

static void sw4lite_rhs4sg_base(float ****rhs,
                                const float ****u,
                                const float ***mu,
                                const float ***lambda, float h) {
  const float w[5] = {-1.0f / 12.0f, 2.0f / 3.0f, 0.0f, -2.0f / 3.0f,
                      1.0f / 12.0f};
  const float inv_h2 = 1.0f / (h * h);

#pragma omp parallel for schedule(static)
  for (int k = POINTS_PER_DIR; k < NZ - POINTS_PER_DIR; ++k) {
    for (int j = POINTS_PER_DIR; j < NY - POINTS_PER_DIR; ++j) {
      for (int i = POINTS_PER_DIR; i < NX - POINTS_PER_DIR; ++i) {
        const float mu_c = mu[i][j][k];
        const float la_c = lambda[i][j][k];

        // Component 0 (x-direction divergence)
        {
          float lap0 = 0.0f;
          for (int offset = -2; offset <= 2; ++offset) {
            lap0 += w[offset + 2] * (u[0][i + offset][j][k] +
                                     u[0][i][j + offset][k] +
                                     u[0][i][j][k + offset]);
          }
          float div_term0 = u[0][i + 1][j][k] - u[0][i - 1][j][k];
          rhs[0][i][j][k] =
              mu_c * lap0 * inv_h2 + (la_c + mu_c) * div_term0 * (0.5f / h);
        }

        // Component 1 (y-direction divergence)
        {
          float lap1 = 0.0f;
          for (int offset = -2; offset <= 2; ++offset) {
            lap1 += w[offset + 2] * (u[1][i + offset][j][k] +
                                     u[1][i][j + offset][k] +
                                     u[1][i][j][k + offset]);
          }
          float div_term1 = u[1][i][j + 1][k] - u[1][i][j - 1][k];
          rhs[1][i][j][k] =
              mu_c * lap1 * inv_h2 + (la_c + mu_c) * div_term1 * (0.5f / h);
        }

        // Component 2 (z-direction divergence)
        {
          float lap2 = 0.0f;
          for (int offset = -2; offset <= 2; ++offset) {
            lap2 += w[offset + 2] * (u[2][i + offset][j][k] +
                                     u[2][i][j + offset][k] +
                                     u[2][i][j][k + offset]);
          }
          float div_term2 = u[2][i][j][k + 1] - u[2][i][j][k - 1];
          rhs[2][i][j][k] =
              mu_c * lap2 * inv_h2 + (la_c + mu_c) * div_term2 * (0.5f / h);
        }
      }
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();

  CARTS_E2E_TIMER_START("sw4lite_rhs4sg_base");

  // Allocate 4D arrays for u and rhs [COMP][NX][NY][NZ]
  float ****u = (float ****)malloc(COMP * sizeof(float ***));
  float ****rhs = (float ****)malloc(COMP * sizeof(float ***));
  // Allocate 3D arrays for mu and lambda [NX][NY][NZ]
  float ***mu = (float ***)malloc(NX * sizeof(float **));
  float ***lambda = (float ***)malloc(NX * sizeof(float **));

  for (int c = 0; c < COMP; ++c) {
    u[c] = (float ***)malloc(NX * sizeof(float **));
    rhs[c] = (float ***)malloc(NX * sizeof(float **));
    for (int i = 0; i < NX; ++i) {
      u[c][i] = (float **)malloc(NY * sizeof(float *));
      rhs[c][i] = (float **)malloc(NY * sizeof(float *));
      for (int j = 0; j < NY; ++j) {
        u[c][i][j] = (float *)malloc(NZ * sizeof(float));
        rhs[c][i][j] = (float *)malloc(NZ * sizeof(float));
      }
    }
  }

  for (int i = 0; i < NX; ++i) {
    mu[i] = (float **)malloc(NY * sizeof(float *));
    lambda[i] = (float **)malloc(NY * sizeof(float *));
    for (int j = 0; j < NY; ++j) {
      mu[i][j] = (float *)malloc(NZ * sizeof(float));
      lambda[i][j] = (float *)malloc(NZ * sizeof(float));
    }
  }

  init_array(u, mu, lambda, rhs);

  // CARTS_KERNEL_TIMER_START("sw4lite_rhs4sg_base");
  sw4lite_rhs4sg_base(rhs, u, mu, lambda, 1.0f);
  // CARTS_KERNEL_TIMER_STOP("sw4lite_rhs4sg_base");

  // Compute checksum using double for better precision with large arrays
  double checksum = 0.0;
  for (int c = 0; c < COMP; ++c) {
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
        for (int k = 0; k < NZ; ++k) {
          checksum += fabs((double)rhs[c][i][j][k]);
        }
      }
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

  // Free arrays
  for (int c = 0; c < COMP; ++c) {
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
        free(u[c][i][j]);
        free(rhs[c][i][j]);
      }
      free(u[c][i]);
      free(rhs[c][i]);
    }
    free(u[c]);
    free(rhs[c]);
  }
  free(u);
  free(rhs);

  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      free(mu[i][j]);
      free(lambda[i][j]);
    }
    free(mu[i]);
    free(lambda[i]);
  }
  free(mu);
  free(lambda);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

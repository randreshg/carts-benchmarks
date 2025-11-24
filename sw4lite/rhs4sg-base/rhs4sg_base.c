#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

#pragma omp parallel for collapse(2) schedule(static)
  for (int k = POINTS_PER_DIR; k < NZ - POINTS_PER_DIR; ++k) {
    for (int j = POINTS_PER_DIR; j < NY - POINTS_PER_DIR; ++j) {
      for (int i = POINTS_PER_DIR; i < NX - POINTS_PER_DIR; ++i) {
        const float mu_c = mu[i][j][k];
        const float la_c = lambda[i][j][k];

        for (int c = 0; c < COMP; ++c) {
          float lap = 0.0f;
          for (int offset = -2; offset <= 2; ++offset) {
            lap += w[offset + 2] * (u[c][i + offset][j][k] +
                                    u[c][i][j + offset][k] +
                                    u[c][i][j][k + offset]);
          }

          float div_term = 0.0f;
          if (c == 0) {
            div_term = u[c][i + 1][j][k] - u[c][i - 1][j][k];
          } else if (c == 1) {
            div_term = u[c][i][j + 1][k] - u[c][i][j - 1][k];
          } else {
            div_term = u[c][i][j][k + 1] - u[c][i][j][k - 1];
          }

          rhs[c][i][j][k] =
              mu_c * lap * inv_h2 + (la_c + mu_c) * div_term * (0.5f / h);
        }
      }
    }
  }
}

static float checksum(const float ****rhs) {
  float sum = 0.0f;
  for (int c = 0; c < COMP; ++c) {
    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
        for (int k = 0; k < NZ; ++k) {
          sum += rhs[c][i][j][k];
        }
      }
    }
  }
  return sum;
}

int main(void) {
  // Allocate 4D arrays for u and rhs [COMP][NX][NY][NZ]
  float ****u = (float ****)malloc(COMP * sizeof(float ***));
  float ****rhs = (float ****)malloc(COMP * sizeof(float ***));
  // Allocate 3D arrays for mu and lambda [NX][NY][NZ]
  float ***mu = (float ***)malloc(NX * sizeof(float **));
  float ***lambda = (float ***)malloc(NX * sizeof(float **));

  if (!u || !mu || !lambda || !rhs) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  for (int c = 0; c < COMP; ++c) {
    u[c] = (float ***)malloc(NX * sizeof(float **));
    rhs[c] = (float ***)malloc(NX * sizeof(float **));
    if (!u[c] || !rhs[c]) {
      fprintf(stderr, "allocation failure\n");
      return 1;
    }
    for (int i = 0; i < NX; ++i) {
      u[c][i] = (float **)malloc(NY * sizeof(float *));
      rhs[c][i] = (float **)malloc(NY * sizeof(float *));
      if (!u[c][i] || !rhs[c][i]) {
        fprintf(stderr, "allocation failure\n");
        return 1;
      }
      for (int j = 0; j < NY; ++j) {
        u[c][i][j] = (float *)malloc(NZ * sizeof(float));
        rhs[c][i][j] = (float *)malloc(NZ * sizeof(float));
        if (!u[c][i][j] || !rhs[c][i][j]) {
          fprintf(stderr, "allocation failure\n");
          return 1;
        }
      }
    }
  }

  for (int i = 0; i < NX; ++i) {
    mu[i] = (float **)malloc(NY * sizeof(float *));
    lambda[i] = (float **)malloc(NY * sizeof(float *));
    if (!mu[i] || !lambda[i]) {
      fprintf(stderr, "allocation failure\n");
      return 1;
    }
    for (int j = 0; j < NY; ++j) {
      mu[i][j] = (float *)malloc(NZ * sizeof(float));
      lambda[i][j] = (float *)malloc(NZ * sizeof(float));
      if (!mu[i][j] || !lambda[i][j]) {
        fprintf(stderr, "allocation failure\n");
        return 1;
      }
    }
  }

  init_array(u, mu, lambda, rhs);
  sw4lite_rhs4sg_base(rhs, u, mu, lambda, 1.0f);

  printf("sw4lite_rhs4sg_base checksum=%f\n", checksum(rhs));

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
  return 0;
}

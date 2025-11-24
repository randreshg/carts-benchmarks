#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NX
#define NX 40
#endif
#ifndef NY
#define NY 40
#endif
#ifndef NZ
#define NZ 40
#endif
#ifndef DT
#define DT 0.001f
#endif

static void init(float ***vx, float ***vy, float ***vz, float ***rho,
                 float ***mu, float ***lambda, float ***sxx, float ***syy,
                 float ***szz, float ***sxy, float ***sxz, float ***syz) {
  int idx = 0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        vx[i][j][k] = 0.001f * (float)(idx % 17);
        vy[i][j][k] = 0.0015f * (float)((idx * 3) % 19);
        vz[i][j][k] = 0.0008f * (float)((idx * 5) % 23);
        rho[i][j][k] = 2500.0f + (float)(idx % 7);
        mu[i][j][k] = 30.0f + 0.05f * (float)(idx % 11);
        lambda[i][j][k] = 20.0f + 0.04f * (float)(idx % 13);
        sxx[i][j][k] = syy[i][j][k] = szz[i][j][k] = sxy[i][j][k] = sxz[i][j][k] = syz[i][j][k] = 0.0f;
        idx++;
      }
    }
  }
}

static inline float derivative(const float ***arr, int i, int j, int k,
                               int dir) {
  switch (dir) {
  case 0:
    return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
  case 1:
    return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
  default:
    return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
  }
}

static void specfem3d_update_stress(
    float ***sxx, float ***syy, float ***szz,
    float ***sxy, float ***sxz, float ***syz,
    const float ***vx, const float ***vy,
    const float ***vz, const float ***mu,
    const float ***lambda) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 2; k < NZ - 2; ++k) {
    for (int j = 2; j < NY - 2; ++j) {
      for (int i = 2; i < NX - 2; ++i) {
        const float mu_c = mu[i][j][k];
        const float la_c = lambda[i][j][k];

        const float dvx_dx = derivative(vx, i, j, k, 0);
        const float dvy_dy = derivative(vy, i, j, k, 1);
        const float dvz_dz = derivative(vz, i, j, k, 2);

        const float trace = dvx_dx + dvy_dy + dvz_dz;
        const float two_mu = 2.0f * mu_c;

        sxx[i][j][k] += DT * (two_mu * dvx_dx + la_c * trace);
        syy[i][j][k] += DT * (two_mu * dvy_dy + la_c * trace);
        szz[i][j][k] += DT * (two_mu * dvz_dz + la_c * trace);

        sxy[i][j][k] += DT * mu_c * (derivative(vx, i, j, k, 1) +
                                     derivative(vy, i, j, k, 0));
        sxz[i][j][k] += DT * mu_c * (derivative(vx, i, j, k, 2) +
                                     derivative(vz, i, j, k, 0));
        syz[i][j][k] += DT * mu_c * (derivative(vy, i, j, k, 2) +
                                     derivative(vz, i, j, k, 1));
      }
    }
  }
}

static float checksum(const float ***arr) {
  float sum = 0.0f;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        sum += arr[i][j][k];
      }
    }
  }
  return sum;
}

int main(void) {
  // Allocate 3D arrays
  float ***vx = (float ***)malloc(NX * sizeof(float **));
  float ***vy = (float ***)malloc(NX * sizeof(float **));
  float ***vz = (float ***)malloc(NX * sizeof(float **));
  float ***rho = (float ***)malloc(NX * sizeof(float **));
  float ***mu = (float ***)malloc(NX * sizeof(float **));
  float ***lambda = (float ***)malloc(NX * sizeof(float **));
  float ***sxx = (float ***)malloc(NX * sizeof(float **));
  float ***syy = (float ***)malloc(NX * sizeof(float **));
  float ***szz = (float ***)malloc(NX * sizeof(float **));
  float ***sxy = (float ***)malloc(NX * sizeof(float **));
  float ***sxz = (float ***)malloc(NX * sizeof(float **));
  float ***syz = (float ***)malloc(NX * sizeof(float **));

  if (!vx || !vy || !vz || !rho || !mu || !lambda || !sxx || !syy || !szz ||
      !sxy || !sxz || !syz) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  for (int i = 0; i < NX; ++i) {
    vx[i] = (float **)malloc(NY * sizeof(float *));
    vy[i] = (float **)malloc(NY * sizeof(float *));
    vz[i] = (float **)malloc(NY * sizeof(float *));
    rho[i] = (float **)malloc(NY * sizeof(float *));
    mu[i] = (float **)malloc(NY * sizeof(float *));
    lambda[i] = (float **)malloc(NY * sizeof(float *));
    sxx[i] = (float **)malloc(NY * sizeof(float *));
    syy[i] = (float **)malloc(NY * sizeof(float *));
    szz[i] = (float **)malloc(NY * sizeof(float *));
    sxy[i] = (float **)malloc(NY * sizeof(float *));
    sxz[i] = (float **)malloc(NY * sizeof(float *));
    syz[i] = (float **)malloc(NY * sizeof(float *));
    if (!vx[i] || !vy[i] || !vz[i] || !rho[i] || !mu[i] || !lambda[i] ||
        !sxx[i] || !syy[i] || !szz[i] || !sxy[i] || !sxz[i] || !syz[i]) {
      fprintf(stderr, "allocation failure\n");
      return 1;
    }
    for (int j = 0; j < NY; ++j) {
      vx[i][j] = (float *)malloc(NZ * sizeof(float));
      vy[i][j] = (float *)malloc(NZ * sizeof(float));
      vz[i][j] = (float *)malloc(NZ * sizeof(float));
      rho[i][j] = (float *)malloc(NZ * sizeof(float));
      mu[i][j] = (float *)malloc(NZ * sizeof(float));
      lambda[i][j] = (float *)malloc(NZ * sizeof(float));
      sxx[i][j] = (float *)malloc(NZ * sizeof(float));
      syy[i][j] = (float *)malloc(NZ * sizeof(float));
      szz[i][j] = (float *)malloc(NZ * sizeof(float));
      sxy[i][j] = (float *)malloc(NZ * sizeof(float));
      sxz[i][j] = (float *)malloc(NZ * sizeof(float));
      syz[i][j] = (float *)malloc(NZ * sizeof(float));
      if (!vx[i][j] || !vy[i][j] || !vz[i][j] || !rho[i][j] || !mu[i][j] ||
          !lambda[i][j] || !sxx[i][j] || !syy[i][j] || !szz[i][j] ||
          !sxy[i][j] || !sxz[i][j] || !syz[i][j]) {
        fprintf(stderr, "allocation failure\n");
        return 1;
      }
    }
  }

  init(vx, vy, vz, rho, mu, lambda, sxx, syy, szz, sxy, sxz, syz);
  specfem3d_update_stress(sxx, syy, szz, sxy, sxz, syz, vx, vy, vz, mu,
                          lambda);

  printf("specfem3d_stress checksum=%f\n",
         checksum(sxx) + checksum(syy) + checksum(szz) +
             checksum(sxy) + checksum(sxz) + checksum(syz));

  // Free 3D arrays
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      free(vx[i][j]);
      free(vy[i][j]);
      free(vz[i][j]);
      free(rho[i][j]);
      free(mu[i][j]);
      free(lambda[i][j]);
      free(sxx[i][j]);
      free(syy[i][j]);
      free(szz[i][j]);
      free(sxy[i][j]);
      free(sxz[i][j]);
      free(syz[i][j]);
    }
    free(vx[i]);
    free(vy[i]);
    free(vz[i]);
    free(rho[i]);
    free(mu[i]);
    free(lambda[i]);
    free(sxx[i]);
    free(syy[i]);
    free(szz[i]);
    free(sxy[i]);
    free(sxz[i]);
    free(syz[i]);
  }
  free(vx);
  free(vy);
  free(vz);
  free(rho);
  free(mu);
  free(lambda);
  free(sxx);
  free(syy);
  free(szz);
  free(sxy);
  free(sxz);
  free(syz);
  return 0;
}

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
#ifndef DT
#define DT 0.0005f
#endif

#define COMP 3

static void init(float ***vx, float ***vy, float ***vz, float ***rho, float ***sxx,
                 float ***syy, float ***szz, float ***sxy, float ***sxz, float ***syz) {
  int idx = 0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        vx[i][j][k] = 0.0f;
        vy[i][j][k] = 0.0f;
        vz[i][j][k] = 0.0f;
        rho[i][j][k] = 2500.0f + (float)(idx % 13);
        sxx[i][j][k] = 0.01f * (float)((idx * 7) % 19);
        syy[i][j][k] = 0.01f * (float)((idx * 11) % 23);
        szz[i][j][k] = 0.01f * (float)((idx * 13) % 17);
        sxy[i][j][k] = 0.005f * (float)((idx * 5) % 29);
        sxz[i][j][k] = 0.004f * (float)((idx * 3) % 31);
        syz[i][j][k] = 0.006f * (float)((idx * 2) % 37);
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

static void sw4lite_vel4sg_update(float ***vx, float ***vy,
                                  float ***vz,
                                  const float ***rho,
                                  const float ***sxx,
                                  const float ***syy,
                                  const float ***szz,
                                  const float ***sxy,
                                  const float ***sxz,
                                  const float ***syz) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 2; k < NZ - 2; ++k) {
    for (int j = 2; j < NY - 2; ++j) {
      for (int i = 2; i < NX - 2; ++i) {
        const float inv_rho = 1.0f / rho[i][j][k];

        const float div_vx = derivative(sxx, i, j, k, 0) +
                             derivative(sxy, i, j, k, 1) +
                             derivative(sxz, i, j, k, 2);
        const float div_vy = derivative(sxy, i, j, k, 0) +
                             derivative(syy, i, j, k, 1) +
                             derivative(syz, i, j, k, 2);
        const float div_vz = derivative(sxz, i, j, k, 0) +
                             derivative(syz, i, j, k, 1) +
                             derivative(szz, i, j, k, 2);

        vx[i][j][k] += DT * inv_rho * div_vx;
        vy[i][j][k] += DT * inv_rho * div_vy;
        vz[i][j][k] += DT * inv_rho * div_vz;
      }
    }
  }
}

static float checksum(const float ***vx, const float ***vy, const float ***vz) {
  float sum = 0.0f;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        sum += vx[i][j][k] + vy[i][j][k] + vz[i][j][k];
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
  float ***sxx = (float ***)malloc(NX * sizeof(float **));
  float ***syy = (float ***)malloc(NX * sizeof(float **));
  float ***szz = (float ***)malloc(NX * sizeof(float **));
  float ***sxy = (float ***)malloc(NX * sizeof(float **));
  float ***sxz = (float ***)malloc(NX * sizeof(float **));
  float ***syz = (float ***)malloc(NX * sizeof(float **));

  if (!vx || !vy || !vz || !rho || !sxx || !syy || !szz || !sxy || !sxz ||
      !syz) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }

  for (int i = 0; i < NX; ++i) {
    vx[i] = (float **)malloc(NY * sizeof(float *));
    vy[i] = (float **)malloc(NY * sizeof(float *));
    vz[i] = (float **)malloc(NY * sizeof(float *));
    rho[i] = (float **)malloc(NY * sizeof(float *));
    sxx[i] = (float **)malloc(NY * sizeof(float *));
    syy[i] = (float **)malloc(NY * sizeof(float *));
    szz[i] = (float **)malloc(NY * sizeof(float *));
    sxy[i] = (float **)malloc(NY * sizeof(float *));
    sxz[i] = (float **)malloc(NY * sizeof(float *));
    syz[i] = (float **)malloc(NY * sizeof(float *));
    if (!vx[i] || !vy[i] || !vz[i] || !rho[i] || !sxx[i] || !syy[i] || !szz[i] ||
        !sxy[i] || !sxz[i] || !syz[i]) {
      fprintf(stderr, "allocation failure\n");
      return 1;
    }
    for (int j = 0; j < NY; ++j) {
      vx[i][j] = (float *)malloc(NZ * sizeof(float));
      vy[i][j] = (float *)malloc(NZ * sizeof(float));
      vz[i][j] = (float *)malloc(NZ * sizeof(float));
      rho[i][j] = (float *)malloc(NZ * sizeof(float));
      sxx[i][j] = (float *)malloc(NZ * sizeof(float));
      syy[i][j] = (float *)malloc(NZ * sizeof(float));
      szz[i][j] = (float *)malloc(NZ * sizeof(float));
      sxy[i][j] = (float *)malloc(NZ * sizeof(float));
      sxz[i][j] = (float *)malloc(NZ * sizeof(float));
      syz[i][j] = (float *)malloc(NZ * sizeof(float));
      if (!vx[i][j] || !vy[i][j] || !vz[i][j] || !rho[i][j] || !sxx[i][j] ||
          !syy[i][j] || !szz[i][j] || !sxy[i][j] || !sxz[i][j] || !syz[i][j]) {
        fprintf(stderr, "allocation failure\n");
        return 1;
      }
    }
  }

  init(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);
  sw4lite_vel4sg_update(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);

  printf("sw4lite_vel4sg_base checksum=%f\n", checksum(vx, vy, vz));

  // Free 3D arrays
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      free(vx[i][j]);
      free(vy[i][j]);
      free(vz[i][j]);
      free(rho[i][j]);
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
  free(sxx);
  free(syy);
  free(szz);
  free(sxy);
  free(sxz);
  free(syz);
  return 0;
}

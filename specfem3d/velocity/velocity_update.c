#include <omp.h>
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
#ifndef DT
#define DT 0.001f
#endif

static void init(float ***vx, float ***vy, float ***vz, float ***rho,
                 float ***sxx, float ***syy, float ***szz, float ***sxy,
                 float ***sxz, float ***syz) {
  int idx = 0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        vx[i][j][k] = 0.0f;
        vy[i][j][k] = 0.0f;
        vz[i][j][k] = 0.0f;
        rho[i][j][k] = 2300.0f + (float)(idx % 11);
        sxx[i][j][k] = 0.02f * (float)((idx * 2) % 17);
        syy[i][j][k] = 0.02f * (float)((idx * 3) % 19);
        szz[i][j][k] = 0.02f * (float)((idx * 5) % 23);
        sxy[i][j][k] = 0.01f * (float)((idx * 7) % 13);
        sxz[i][j][k] = 0.01f * (float)((idx * 11) % 29);
        syz[i][j][k] = 0.01f * (float)((idx * 13) % 31);
        idx++;
      }
    }
  }
}

static inline float diff_x(const float ***arr, int i, int j, int k) {
  return arr[i + 1][j][k] - arr[i][j][k];
}

static inline float diff_y(const float ***arr, int i, int j, int k) {
  return arr[i][j + 1][k] - arr[i][j][k];
}

static inline float diff_z(const float ***arr, int i, int j, int k) {
  return arr[i][j][k + 1] - arr[i][j][k];
}

static void specfem_velocity_update(float ***vx, float ***vy, float ***vz,
                                    const float ***rho, const float ***sxx,
                                    const float ***syy, const float ***szz,
                                    const float ***sxy, const float ***sxz,
                                    const float ***syz) {
#pragma omp parallel for schedule(static)
  for (int k = 1; k < NZ - 1; ++k) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int i = 1; i < NX - 1; ++i) {
        const float inv_rho = 1.0f / rho[i][j][k];

        const float dvx =
            diff_x(sxx, i, j, k) + diff_y(sxy, i, j, k) + diff_z(sxz, i, j, k);
        const float dvy =
            diff_x(sxy, i, j, k) + diff_y(syy, i, j, k) + diff_z(syz, i, j, k);
        const float dvz =
            diff_x(sxz, i, j, k) + diff_y(syz, i, j, k) + diff_z(szz, i, j, k);

        vx[i][j][k] += DT * inv_rho * dvx;
        vy[i][j][k] += DT * inv_rho * dvy;
        vz[i][j][k] += DT * inv_rho * dvz;
      }
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();

  CARTS_E2E_TIMER_START("specfem_velocity_update");

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
    }
  }

  init(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);

  CARTS_KERNEL_TIMER_START("specfem_velocity_update");
  specfem_velocity_update(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);
  CARTS_KERNEL_TIMER_STOP("specfem_velocity_update");

  CARTS_E2E_TIMER_STOP();

  // Compute checksum inline
  float checksum = 0.0f;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        checksum += vx[i][j][k] + vy[i][j][k] + vz[i][j][k];
      }
    }
  }
  CARTS_BENCH_CHECKSUM(checksum);

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

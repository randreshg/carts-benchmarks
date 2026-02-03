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
#define DT 0.001
#endif

static void init(double ***vx, double ***vy, double ***vz, double ***rho,
                 double ***sxx, double ***syy, double ***szz, double ***sxy,
                 double ***sxz, double ***syz) {
  int idx = 0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        vx[i][j][k] = 0.0;
        vy[i][j][k] = 0.0;
        vz[i][j][k] = 0.0;
        rho[i][j][k] = 2300.0 + (double)(idx % 11);
        sxx[i][j][k] = 0.02 * (double)((idx * 2) % 17);
        syy[i][j][k] = 0.02 * (double)((idx * 3) % 19);
        szz[i][j][k] = 0.02 * (double)((idx * 5) % 23);
        sxy[i][j][k] = 0.01 * (double)((idx * 7) % 13);
        sxz[i][j][k] = 0.01 * (double)((idx * 11) % 29);
        syz[i][j][k] = 0.01 * (double)((idx * 13) % 31);
        idx++;
      }
    }
  }
}

static inline double diff_x(const double ***arr, int i, int j, int k) {
  return arr[i + 1][j][k] - arr[i][j][k];
}

static inline double diff_y(const double ***arr, int i, int j, int k) {
  return arr[i][j + 1][k] - arr[i][j][k];
}

static inline double diff_z(const double ***arr, int i, int j, int k) {
  return arr[i][j][k + 1] - arr[i][j][k];
}

static void specfem_velocity_update(double ***vx, double ***vy, double ***vz,
                                    const double ***rho, const double ***sxx,
                                    const double ***syy, const double ***szz,
                                    const double ***sxy, const double ***sxz,
                                    const double ***syz) {
#pragma omp parallel for schedule(static)
  for (int k = 1; k < NZ - 1; ++k) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int i = 1; i < NX - 1; ++i) {
        const double inv_rho = 1.0 / rho[i][j][k];

        const double dvx =
            diff_x(sxx, i, j, k) + diff_y(sxy, i, j, k) + diff_z(sxz, i, j, k);
        const double dvy =
            diff_x(sxy, i, j, k) + diff_y(syy, i, j, k) + diff_z(syz, i, j, k);
        const double dvz =
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
  double ***vx = (double ***)malloc(NX * sizeof(double **));
  double ***vy = (double ***)malloc(NX * sizeof(double **));
  double ***vz = (double ***)malloc(NX * sizeof(double **));
  double ***rho = (double ***)malloc(NX * sizeof(double **));
  double ***sxx = (double ***)malloc(NX * sizeof(double **));
  double ***syy = (double ***)malloc(NX * sizeof(double **));
  double ***szz = (double ***)malloc(NX * sizeof(double **));
  double ***sxy = (double ***)malloc(NX * sizeof(double **));
  double ***sxz = (double ***)malloc(NX * sizeof(double **));
  double ***syz = (double ***)malloc(NX * sizeof(double **));

  for (int i = 0; i < NX; ++i) {
    vx[i] = (double **)malloc(NY * sizeof(double *));
    vy[i] = (double **)malloc(NY * sizeof(double *));
    vz[i] = (double **)malloc(NY * sizeof(double *));
    rho[i] = (double **)malloc(NY * sizeof(double *));
    sxx[i] = (double **)malloc(NY * sizeof(double *));
    syy[i] = (double **)malloc(NY * sizeof(double *));
    szz[i] = (double **)malloc(NY * sizeof(double *));
    sxy[i] = (double **)malloc(NY * sizeof(double *));
    sxz[i] = (double **)malloc(NY * sizeof(double *));
    syz[i] = (double **)malloc(NY * sizeof(double *));
    for (int j = 0; j < NY; ++j) {
      vx[i][j] = (double *)malloc(NZ * sizeof(double));
      vy[i][j] = (double *)malloc(NZ * sizeof(double));
      vz[i][j] = (double *)malloc(NZ * sizeof(double));
      rho[i][j] = (double *)malloc(NZ * sizeof(double));
      sxx[i][j] = (double *)malloc(NZ * sizeof(double));
      syy[i][j] = (double *)malloc(NZ * sizeof(double));
      szz[i][j] = (double *)malloc(NZ * sizeof(double));
      sxy[i][j] = (double *)malloc(NZ * sizeof(double));
      sxz[i][j] = (double *)malloc(NZ * sizeof(double));
      syz[i][j] = (double *)malloc(NZ * sizeof(double));
    }
  }

  init(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);

  // CARTS_KERNEL_TIMER_START("specfem_velocity_update");
  specfem_velocity_update(vx, vy, vz, rho, sxx, syy, szz, sxy, sxz, syz);
  // CARTS_KERNEL_TIMER_STOP("specfem_velocity_update");

  // Compute checksum
  double checksum = 0.0;
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

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

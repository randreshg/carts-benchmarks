#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

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
#define DT 0.001
#endif

static void init(double ***vx, double ***vy, double ***vz, double ***rho,
                 double ***mu, double ***lambda, double ***sxx, double ***syy,
                 double ***szz, double ***sxy, double ***sxz, double ***syz) {
  int idx = 0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        vx[i][j][k] = 0.001 * (double)(idx % 17);
        vy[i][j][k] = 0.0015 * (double)((idx * 3) % 19);
        vz[i][j][k] = 0.0008 * (double)((idx * 5) % 23);
        rho[i][j][k] = 2500.0 + (double)(idx % 7);
        mu[i][j][k] = 30.0 + 0.05 * (double)(idx % 11);
        lambda[i][j][k] = 20.0 + 0.04 * (double)(idx % 13);
        sxx[i][j][k] = syy[i][j][k] = szz[i][j][k] = sxy[i][j][k] =
            sxz[i][j][k] = syz[i][j][k] = 0.0;
        idx++;
      }
    }
  }
}

static inline double derivative_x(const double ***arr, int i, int j, int k) {
  return 0.5 * (arr[i + 1][j][k] - arr[i - 1][j][k]);
}

static inline double derivative_y(const double ***arr, int i, int j, int k) {
  return 0.5 * (arr[i][j + 1][k] - arr[i][j - 1][k]);
}

static inline double derivative_z(const double ***arr, int i, int j, int k) {
  return 0.5 * (arr[i][j][k + 1] - arr[i][j][k - 1]);
}

static void specfem3d_update_stress(double ***sxx, double ***syy, double ***szz,
                                    double ***sxy, double ***sxz, double ***syz,
                                    const double ***vx, const double ***vy,
                                    const double ***vz, const double ***mu,
                                    const double ***lambda) {
#pragma omp parallel for schedule(static)
  for (int k = 2; k < NZ - 2; ++k) {
    for (int j = 2; j < NY - 2; ++j) {
      for (int i = 2; i < NX - 2; ++i) {
        const double mu_c = mu[i][j][k];
        const double la_c = lambda[i][j][k];

        const double dvx_dx = derivative_x(vx, i, j, k);
        const double dvy_dy = derivative_y(vy, i, j, k);
        const double dvz_dz = derivative_z(vz, i, j, k);

        const double trace = dvx_dx + dvy_dy + dvz_dz;
        const double two_mu = 2.0 * mu_c;

        sxx[i][j][k] += DT * (two_mu * dvx_dx + la_c * trace);
        syy[i][j][k] += DT * (two_mu * dvy_dy + la_c * trace);
        szz[i][j][k] += DT * (two_mu * dvz_dz + la_c * trace);

        sxy[i][j][k] +=
            DT * mu_c * (derivative_y(vx, i, j, k) + derivative_x(vy, i, j, k));
        sxz[i][j][k] +=
            DT * mu_c * (derivative_z(vx, i, j, k) + derivative_x(vz, i, j, k));
        syz[i][j][k] +=
            DT * mu_c * (derivative_z(vy, i, j, k) + derivative_y(vz, i, j, k));
      }
    }
  }
}

int main(void) {
  CARTS_BENCHMARKS_START();

  CARTS_E2E_TIMER_START("specfem3d_update_stress");

  // Allocate 3D arrays
  double ***vx = (double ***)malloc(NX * sizeof(double **));
  double ***vy = (double ***)malloc(NX * sizeof(double **));
  double ***vz = (double ***)malloc(NX * sizeof(double **));
  double ***rho = (double ***)malloc(NX * sizeof(double **));
  double ***mu = (double ***)malloc(NX * sizeof(double **));
  double ***lambda = (double ***)malloc(NX * sizeof(double **));
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
    mu[i] = (double **)malloc(NY * sizeof(double *));
    lambda[i] = (double **)malloc(NY * sizeof(double *));
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
      mu[i][j] = (double *)malloc(NZ * sizeof(double));
      lambda[i][j] = (double *)malloc(NZ * sizeof(double));
      sxx[i][j] = (double *)malloc(NZ * sizeof(double));
      syy[i][j] = (double *)malloc(NZ * sizeof(double));
      szz[i][j] = (double *)malloc(NZ * sizeof(double));
      sxy[i][j] = (double *)malloc(NZ * sizeof(double));
      sxz[i][j] = (double *)malloc(NZ * sizeof(double));
      syz[i][j] = (double *)malloc(NZ * sizeof(double));
    }
  }

  init(vx, vy, vz, rho, mu, lambda, sxx, syy, szz, sxy, sxz, syz);

  // CARTS_KERNEL_TIMER_START("specfem3d_update_stress");
  specfem3d_update_stress(sxx, syy, szz, sxy, sxz, syz, vx, vy, vz, mu, lambda);
  // CARTS_KERNEL_TIMER_STOP("specfem3d_update_stress");

  // Compute checksum
  double checksum = 0.0;
  for (int i = 0; i < NX; ++i) {
    for (int j = 0; j < NY; ++j) {
      for (int k = 0; k < NZ; ++k) {
        checksum += sxx[i][j][k] + syy[i][j][k] + szz[i][j][k] +
                    sxy[i][j][k] + sxz[i][j][k] + syz[i][j][k];
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

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();
  return 0;
}

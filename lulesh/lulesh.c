/*
   This is a CARTS-compatible C version of LULESH 2.0
   Converted from C++ to plain C with carts-compatible patterns

                  Copyright (c) 2010-2013.
       Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory.
                   LLNL-CODE-461231
                 All rights reserved.

 This file is part of LULESH, Version 2.0.
 Please also read this link -- http://www.opensource.org/licenses/index.php

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:

  * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the disclaimer below.

  * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the disclaimer (as noted below)
      in the documentation and/or other materials provided with the
      distribution.

  * Neither the name of the LLNS/LLNL nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
 THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <limits.h>
#include <math.h>
#if _OPENMP
#include <omp.h>
#endif
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

//**************************************************
// Allow flexibility for arithmetic representations
//**************************************************

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Precision specification
typedef float real4;
typedef double real8;
typedef long double real10;

typedef int Index_t;
typedef real8 Real_t;
typedef int Int_t;

enum { VolumeError = -1, QStopError = -2 };

// Math functions - use appropriate types
#define SQRT(x) sqrt(x)
#define CBRT(x) cbrt(x)
#define FABS(x) fabs(x)

// Boundary condition flags
#define XI_M 0x00007
#define XI_M_SYMM 0x00001
#define XI_M_FREE 0x00002
#define XI_M_COMM 0x00004

#define XI_P 0x00038
#define XI_P_SYMM 0x00008
#define XI_P_FREE 0x00010
#define XI_P_COMM 0x00020

#define ETA_M 0x001c0
#define ETA_M_SYMM 0x00040
#define ETA_M_FREE 0x00080
#define ETA_M_COMM 0x00100

#define ETA_P 0x00e00
#define ETA_P_SYMM 0x00200
#define ETA_P_FREE 0x00400
#define ETA_P_COMM 0x00800

#define ZETA_M 0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P 0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))
#define CACHE_ALIGN_REAL(n)                                                    \
  (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL - 1))

//**************************************************
// Constants (formerly in Domain struct)
//**************************************************

static const Real_t c_e_cut = 1.0e-7;
static const Real_t c_p_cut = 1.0e-7;
static const Real_t c_q_cut = 1.0e-7;
static const Real_t c_v_cut = 1.0e-10;
static const Real_t c_u_cut = 1.0e-7;
static const Real_t c_hgcoef = 3.0;
static const Real_t c_ss4o3 = 1.3333333333333333;
#ifndef CARTS_QSTOP
#define CARTS_QSTOP 1.0e+12
#endif
static const Real_t c_qstop = CARTS_QSTOP;
static const Real_t c_monoq_max_slope = 1.0;
static const Real_t c_monoq_limiter_mult = 2.0;
static const Real_t c_qlc_monoq = 0.5;
static const Real_t c_qqc_monoq = 0.6666666666666666;
static const Real_t c_qqc = 2.0;
static const Real_t c_eosvmax = 1.0e+9;
static const Real_t c_eosvmin = 1.0e-9;
static const Real_t c_pmin = 0.0;
static const Real_t c_emin = -1.0e+15;
static const Real_t c_dvovmax = 0.1;
static const Real_t c_refdens = 1.0;

//**************************************************
// Command line options
//**************************************************

typedef struct cmdLineOpts {
  Int_t its;
  Int_t nx;
  Int_t numReg;
  Int_t showProg;
  Int_t quiet;
  Int_t cost;
  Int_t balance;
} cmdLineOpts;

//**************************************************
// Function prototypes
//**************************************************

static inline Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8],
                                    const Real_t z[8]);

//**************************************************
// Helper allocation functions
//**************************************************

static inline Real_t *AllocateReal(size_t size) {
  return (Real_t *)malloc(sizeof(Real_t) * size);
}

static inline Index_t *AllocateIndex(size_t size) {
  return (Index_t *)malloc(sizeof(Index_t) * size);
}

static inline Int_t *AllocateInt(size_t size) {
  return (Int_t *)malloc(sizeof(Int_t) * size);
}

//**************************************************
// 2D Array allocation helpers (CARTS-compatible)
//**************************************************

static inline Index_t **AllocateIndex2D(size_t rows, size_t cols) {
  Index_t **arr = (Index_t **)malloc(rows * sizeof(Index_t *));
  for (size_t i = 0; i < rows; i++) {
    arr[i] = (Index_t *)malloc(cols * sizeof(Index_t));
  }
  return arr;
}

static inline void FreeIndex2D(Index_t **arr, size_t rows) {
  for (size_t i = 0; i < rows; i++) {
    free(arr[i]);
  }
  free(arr);
}

static inline Real_t **AllocateReal2D(size_t rows, size_t cols) {
  Real_t **arr = (Real_t **)malloc(rows * sizeof(Real_t *));
  for (size_t i = 0; i < rows; i++) {
    arr[i] = (Real_t *)malloc(cols * sizeof(Real_t));
  }
  return arr;
}

static inline void FreeReal2D(Real_t **arr, size_t rows) {
  for (size_t i = 0; i < rows; i++) {
    free(arr[i]);
  }
  free(arr);
}

//**************************************************
// Time increment
//**************************************************

static inline void TimeIncrement(Real_t *deltatime, Real_t *time, Int_t *cycle,
                                 Real_t stoptime, Real_t dtfixed,
                                 Real_t dtcourant, Real_t dthydro,
                                 Real_t deltatimemultlb, Real_t deltatimemultub,
                                 Real_t dtmax) {
  Real_t targetdt = stoptime - (*time);

  if ((dtfixed <= 0.0) && ((*cycle) != 0)) {
    Real_t ratio;
    Real_t olddt = *deltatime;
    Real_t gnewdt = 1.0e+20;
    Real_t newdt;

    if (dtcourant < gnewdt) {
      gnewdt = dtcourant / 2.0;
    }
    if (dthydro < gnewdt) {
      gnewdt = dthydro * 2.0 / 3.0;
    }

    newdt = gnewdt;
    ratio = newdt / olddt;

    if (ratio >= 1.0) {
      if (ratio < deltatimemultlb) {
        newdt = olddt;
      } else if (ratio > deltatimemultub) {
        newdt = olddt * deltatimemultub;
      }
    }

    if (newdt > dtmax) {
      newdt = dtmax;
    }
    *deltatime = newdt;
  }

  if ((targetdt > (*deltatime)) && (targetdt < (4.0 * (*deltatime) / 3.0))) {
    targetdt = 2.0 * (*deltatime) / 3.0;
  }

  if (targetdt < (*deltatime)) {
    *deltatime = targetdt;
  }

  *time += *deltatime;
  ++(*cycle);
}

//**************************************************
// Collect nodes to element nodes (inline gather)
//**************************************************

static inline void CollectNodesToElemNodes(const Real_t *x, const Real_t *y,
                                           const Real_t *z, Index_t *elemToNode,
                                           Real_t elemX[8], Real_t elemY[8],
                                           Real_t elemZ[8]) {
  Index_t nd0i = elemToNode[0];
  Index_t nd1i = elemToNode[1];
  Index_t nd2i = elemToNode[2];
  Index_t nd3i = elemToNode[3];
  Index_t nd4i = elemToNode[4];
  Index_t nd5i = elemToNode[5];
  Index_t nd6i = elemToNode[6];
  Index_t nd7i = elemToNode[7];

  elemX[0] = x[nd0i];
  elemX[1] = x[nd1i];
  elemX[2] = x[nd2i];
  elemX[3] = x[nd3i];
  elemX[4] = x[nd4i];
  elemX[5] = x[nd5i];
  elemX[6] = x[nd6i];
  elemX[7] = x[nd7i];

  elemY[0] = y[nd0i];
  elemY[1] = y[nd1i];
  elemY[2] = y[nd2i];
  elemY[3] = y[nd3i];
  elemY[4] = y[nd4i];
  elemY[5] = y[nd5i];
  elemY[6] = y[nd6i];
  elemY[7] = y[nd7i];

  elemZ[0] = z[nd0i];
  elemZ[1] = z[nd1i];
  elemZ[2] = z[nd2i];
  elemZ[3] = z[nd3i];
  elemZ[4] = z[nd4i];
  elemZ[5] = z[nd5i];
  elemZ[6] = z[nd6i];
  elemZ[7] = z[nd7i];
}

//**************************************************
// Initialize stress terms for elements
//**************************************************

static inline void InitStressTermsForElems(const Real_t *p, const Real_t *q,
                                           Real_t *sigxx, Real_t *sigyy,
                                           Real_t *sigzz, Index_t numElem) {
#pragma omp parallel for firstprivate(numElem)
  for (Index_t i = 0; i < numElem; ++i) {
    sigxx[i] = sigyy[i] = sigzz[i] = -p[i] - q[i];
  }
}

//**************************************************
// Calculate element shape function derivatives
//**************************************************

static inline void CalcElemShapeFunctionDerivatives(Real_t const x[],
                                                    Real_t const y[],
                                                    Real_t const z[],
                                                    Real_t b[][8],
                                                    Real_t *const volume) {
  const Real_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
  const Real_t x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
  const Real_t y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3];
  const Real_t y4 = y[4], y5 = y[5], y6 = y[6], y7 = y[7];
  const Real_t z0 = z[0], z1 = z[1], z2 = z[2], z3 = z[3];
  const Real_t z4 = z[4], z5 = z[5], z6 = z[6], z7 = z[7];

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = 0.125 * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
  fjxet = 0.125 * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
  fjxze = 0.125 * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));

  fjyxi = 0.125 * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
  fjyet = 0.125 * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
  fjyze = 0.125 * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));

  fjzxi = 0.125 * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
  fjzet = 0.125 * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
  fjzze = 0.125 * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));

  cjxxi = (fjyet * fjzze) - (fjzet * fjyze);
  cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze = (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi = -(fjxet * fjzze) + (fjzet * fjxze);
  cjyet = (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi = (fjxet * fjyze) - (fjyet * fjxze);
  cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze = (fjxxi * fjyet) - (fjyxi * fjxet);

  b[0][0] = -cjxxi - cjxet - cjxze;
  b[0][1] = cjxxi - cjxet - cjxze;
  b[0][2] = cjxxi + cjxet - cjxze;
  b[0][3] = -cjxxi + cjxet - cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] = -cjyxi - cjyet - cjyze;
  b[1][1] = cjyxi - cjyet - cjyze;
  b[1][2] = cjyxi + cjyet - cjyze;
  b[1][3] = -cjyxi + cjyet - cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] = -cjzxi - cjzet - cjzze;
  b[2][1] = cjzxi - cjzet - cjzze;
  b[2][2] = cjzxi + cjzet - cjzze;
  b[2][3] = -cjzxi + cjzet - cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  *volume = 8.0 * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

//**************************************************
// Sum element face normal
//**************************************************

static inline void SumElemFaceNormal(
    Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0, Real_t *normalX1,
    Real_t *normalY1, Real_t *normalZ1, Real_t *normalX2, Real_t *normalY2,
    Real_t *normalZ2, Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
    const Real_t x0, const Real_t y0, const Real_t z0, const Real_t x1,
    const Real_t y1, const Real_t z1, const Real_t x2, const Real_t y2,
    const Real_t z2, const Real_t x3, const Real_t y3, const Real_t z3) {
  Real_t bisectX0 = 0.5 * (x3 + x2 - x1 - x0);
  Real_t bisectY0 = 0.5 * (y3 + y2 - y1 - y0);
  Real_t bisectZ0 = 0.5 * (z3 + z2 - z1 - z0);
  Real_t bisectX1 = 0.5 * (x2 + x1 - x3 - x0);
  Real_t bisectY1 = 0.5 * (y2 + y1 - y3 - y0);
  Real_t bisectZ1 = 0.5 * (z2 + z1 - z3 - z0);
  Real_t areaX = 0.25 * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
  Real_t areaY = 0.25 * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
  Real_t areaZ = 0.25 * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

  *normalX0 += areaX;
  *normalX1 += areaX;
  *normalX2 += areaX;
  *normalX3 += areaX;

  *normalY0 += areaY;
  *normalY1 += areaY;
  *normalY2 += areaY;
  *normalY3 += areaY;

  *normalZ0 += areaZ;
  *normalZ1 += areaZ;
  *normalZ2 += areaZ;
  *normalZ3 += areaZ;
}

//**************************************************
// Calculate element node normals
//**************************************************

static inline void CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8],
                                       Real_t pfz[8], const Real_t x[8],
                                       const Real_t y[8], const Real_t z[8]) {
  for (Index_t i = 0; i < 8; ++i) {
    pfx[i] = 0.0;
    pfy[i] = 0.0;
    pfz[i] = 0.0;
  }

  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0], &pfx[1], &pfy[1], &pfz[1],
                    &pfx[2], &pfy[2], &pfz[2], &pfx[3], &pfy[3], &pfz[3], x[0],
                    y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2], x[3], y[3],
                    z[3]);

  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0], &pfx[4], &pfy[4], &pfz[4],
                    &pfx[5], &pfy[5], &pfz[5], &pfx[1], &pfy[1], &pfz[1], x[0],
                    y[0], z[0], x[4], y[4], z[4], x[5], y[5], z[5], x[1], y[1],
                    z[1]);

  SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1], &pfx[5], &pfy[5], &pfz[5],
                    &pfx[6], &pfy[6], &pfz[6], &pfx[2], &pfy[2], &pfz[2], x[1],
                    y[1], z[1], x[5], y[5], z[5], x[6], y[6], z[6], x[2], y[2],
                    z[2]);

  SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2], &pfx[6], &pfy[6], &pfz[6],
                    &pfx[7], &pfy[7], &pfz[7], &pfx[3], &pfy[3], &pfz[3], x[2],
                    y[2], z[2], x[6], y[6], z[6], x[7], y[7], z[7], x[3], y[3],
                    z[3]);

  SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3], &pfx[7], &pfy[7], &pfz[7],
                    &pfx[4], &pfy[4], &pfz[4], &pfx[0], &pfy[0], &pfz[0], x[3],
                    y[3], z[3], x[7], y[7], z[7], x[4], y[4], z[4], x[0], y[0],
                    z[0]);

  SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4], &pfx[7], &pfy[7], &pfz[7],
                    &pfx[6], &pfy[6], &pfz[6], &pfx[5], &pfy[5], &pfz[5], x[4],
                    y[4], z[4], x[7], y[7], z[7], x[6], y[6], z[6], x[5], y[5],
                    z[5]);
}

//**************************************************
// Sum element stresses to node forces
//**************************************************

static inline void
SumElemStressesToNodeForces(const Real_t B[][8], const Real_t stress_xx,
                            const Real_t stress_yy, const Real_t stress_zz,
                            Real_t fx[], Real_t fy[], Real_t fz[]) {
  for (Index_t i = 0; i < 8; i++) {
    fx[i] = -(stress_xx * B[0][i]);
    fy[i] = -(stress_yy * B[1][i]);
    fz[i] = -(stress_zz * B[2][i]);
  }
}

//**************************************************
// Integrate stress for elements
//**************************************************

static inline void IntegrateStressForElems(
    const Real_t *x, const Real_t *y, const Real_t *z, Index_t **nodelist,
    const Index_t *nodeElemStart, const Index_t *nodeElemCornerList, Real_t *fx,
    Real_t *fy, Real_t *fz, Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
    Real_t *determ, Index_t numElem, Index_t numNode) {
  // Simplified for CARTS: always use multi-threaded path
  Real_t **fx_elem = AllocateReal2D(numElem, 8);
  Real_t **fy_elem = AllocateReal2D(numElem, 8);
  Real_t **fz_elem = AllocateReal2D(numElem, 8);

#pragma omp parallel for firstprivate(numElem)
  for (Index_t k = 0; k < numElem; ++k) {
    Real_t B[3][8];
    Real_t x_local[8];
    Real_t y_local[8];
    Real_t z_local[8];

    CollectNodesToElemNodes(x, y, z, nodelist[k], x_local, y_local, z_local);

    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &determ[k]);

    CalcElemNodeNormals(B[0], B[1], B[2], x_local, y_local, z_local);

    SumElemStressesToNodeForces(B, sigxx[k], sigyy[k], sigzz[k], fx_elem[k],
                                fy_elem[k], fz_elem[k]);
  }

#pragma omp parallel for firstprivate(numNode)
  for (Index_t gnode = 0; gnode < numNode; ++gnode) {
    Index_t count = nodeElemStart[gnode + 1] - nodeElemStart[gnode];
    Index_t *cornerList = (Index_t *)&nodeElemCornerList[nodeElemStart[gnode]];
    Real_t fx_tmp = 0.0;
    Real_t fy_tmp = 0.0;
    Real_t fz_tmp = 0.0;
    for (Index_t i = 0; i < count; ++i) {
      Index_t elem = cornerList[i];
      fx_tmp += fx_elem[elem / 8][elem % 8];
      fy_tmp += fy_elem[elem / 8][elem % 8];
      fz_tmp += fz_elem[elem / 8][elem % 8];
    }
    fx[gnode] = fx_tmp;
    fy[gnode] = fy_tmp;
    fz[gnode] = fz_tmp;
  }
  FreeReal2D(fz_elem, numElem);
  FreeReal2D(fy_elem, numElem);
  FreeReal2D(fx_elem, numElem);
}

//**************************************************
// Volume derivative
//**************************************************

static inline void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
                           const Real_t x3, const Real_t x4, const Real_t x5,
                           const Real_t y0, const Real_t y1, const Real_t y2,
                           const Real_t y3, const Real_t y4, const Real_t y5,
                           const Real_t z0, const Real_t z1, const Real_t z2,
                           const Real_t z3, const Real_t z4, const Real_t z5,
                           Real_t *dvdx, Real_t *dvdy, Real_t *dvdz) {
  const Real_t twelfth = 1.0 / 12.0;

  *dvdx = (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
          (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
          (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);

  *dvdy = -(x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
          (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
          (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

  *dvdz = -(y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
          (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
          (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

  *dvdx *= twelfth;
  *dvdy *= twelfth;
  *dvdz *= twelfth;
}

//**************************************************
// Calculate element volume derivative
//**************************************************

static inline void CalcElemVolumeDerivative(Real_t dvdx[8], Real_t dvdy[8],
                                            Real_t dvdz[8], const Real_t x[8],
                                            const Real_t y[8],
                                            const Real_t z[8]) {
  VoluDer(x[1], x[2], x[3], x[4], x[5], x[7], y[1], y[2], y[3], y[4], y[5],
          y[7], z[1], z[2], z[3], z[4], z[5], z[7], &dvdx[0], &dvdy[0],
          &dvdz[0]);
  VoluDer(x[0], x[1], x[2], x[7], x[4], x[6], y[0], y[1], y[2], y[7], y[4],
          y[6], z[0], z[1], z[2], z[7], z[4], z[6], &dvdx[3], &dvdy[3],
          &dvdz[3]);
  VoluDer(x[3], x[0], x[1], x[6], x[7], x[5], y[3], y[0], y[1], y[6], y[7],
          y[5], z[3], z[0], z[1], z[6], z[7], z[5], &dvdx[2], &dvdy[2],
          &dvdz[2]);
  VoluDer(x[2], x[3], x[0], x[5], x[6], x[4], y[2], y[3], y[0], y[5], y[6],
          y[4], z[2], z[3], z[0], z[5], z[6], z[4], &dvdx[1], &dvdy[1],
          &dvdz[1]);
  VoluDer(x[7], x[6], x[5], x[0], x[3], x[1], y[7], y[6], y[5], y[0], y[3],
          y[1], z[7], z[6], z[5], z[0], z[3], z[1], &dvdx[4], &dvdy[4],
          &dvdz[4]);
  VoluDer(x[4], x[7], x[6], x[1], x[0], x[2], y[4], y[7], y[6], y[1], y[0],
          y[2], z[4], z[7], z[6], z[1], z[0], z[2], &dvdx[5], &dvdy[5],
          &dvdz[5]);
  VoluDer(x[5], x[4], x[7], x[2], x[1], x[3], y[5], y[4], y[7], y[2], y[1],
          y[3], z[5], z[4], z[7], z[2], z[1], z[3], &dvdx[6], &dvdy[6],
          &dvdz[6]);
  VoluDer(x[6], x[5], x[4], x[3], x[2], x[0], y[6], y[5], y[4], y[3], y[2],
          y[0], z[6], z[5], z[4], z[3], z[2], z[0], &dvdx[7], &dvdy[7],
          &dvdz[7]);
}

//**************************************************
// Calculate element FB hourglass force
//**************************************************

static inline void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                                            Real_t hourgam[][4],
                                            Real_t coefficient, Real_t *hgfx,
                                            Real_t *hgfy, Real_t *hgfz) {
  Real_t hxx[4];

  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
             hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
             hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
             hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
  }
  for (Index_t i = 0; i < 8; i++) {
    hgfx[i] = coefficient * (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                             hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }

  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
             hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
             hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
             hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
  }
  for (Index_t i = 0; i < 8; i++) {
    hgfy[i] = coefficient * (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                             hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }

  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
             hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
             hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
             hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
  }
  for (Index_t i = 0; i < 8; i++) {
    hgfz[i] = coefficient * (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                             hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
}

//**************************************************
// Calculate FB hourglass force for elements
//**************************************************

static inline void CalcFBHourglassForceForElems(
    const Real_t *xd, const Real_t *yd, const Real_t *zd, Index_t **nodelist,
    const Real_t *ss, const Real_t *elemMass, const Index_t *nodeElemStart,
    const Index_t *nodeElemCornerList, Real_t *fx, Real_t *fy, Real_t *fz,
    Real_t *determ, Real_t *x8n, Real_t *y8n, Real_t *z8n, Real_t *dvdx,
    Real_t *dvdy, Real_t *dvdz, Real_t hourg, Index_t numElem,
    Index_t numNode) {

  // Simplified for CARTS: always use multi-threaded path
  Real_t **fx_elem = AllocateReal2D(numElem, 8);
  Real_t **fy_elem = AllocateReal2D(numElem, 8);
  Real_t **fz_elem = AllocateReal2D(numElem, 8);

  Real_t gamma[4][8];
  gamma[0][0] = 1.;
  gamma[0][1] = 1.;
  gamma[0][2] = -1.;
  gamma[0][3] = -1.;
  gamma[0][4] = -1.;
  gamma[0][5] = -1.;
  gamma[0][6] = 1.;
  gamma[0][7] = 1.;
  gamma[1][0] = 1.;
  gamma[1][1] = -1.;
  gamma[1][2] = -1.;
  gamma[1][3] = 1.;
  gamma[1][4] = -1.;
  gamma[1][5] = 1.;
  gamma[1][6] = 1.;
  gamma[1][7] = -1.;
  gamma[2][0] = 1.;
  gamma[2][1] = -1.;
  gamma[2][2] = 1.;
  gamma[2][3] = -1.;
  gamma[2][4] = 1.;
  gamma[2][5] = -1.;
  gamma[2][6] = 1.;
  gamma[2][7] = -1.;
  gamma[3][0] = -1.;
  gamma[3][1] = 1.;
  gamma[3][2] = -1.;
  gamma[3][3] = 1.;
  gamma[3][4] = 1.;
  gamma[3][5] = -1.;
  gamma[3][6] = 1.;
  gamma[3][7] = -1.;

  for (Index_t i2 = 0; i2 < numElem; ++i2) {
    Real_t hgfx[8], hgfy[8], hgfz[8];
    Real_t coefficient;
    Real_t hourgam[8][4];
    Real_t xd1[8], yd1[8], zd1[8];

    Index_t i3 = 8 * i2;
    Real_t volinv = 1.0 / determ[i2];

    for (Index_t i1 = 0; i1 < 4; ++i1) {
      Real_t hourmodx =
          x8n[i3] * gamma[i1][0] + x8n[i3 + 1] * gamma[i1][1] +
          x8n[i3 + 2] * gamma[i1][2] + x8n[i3 + 3] * gamma[i1][3] +
          x8n[i3 + 4] * gamma[i1][4] + x8n[i3 + 5] * gamma[i1][5] +
          x8n[i3 + 6] * gamma[i1][6] + x8n[i3 + 7] * gamma[i1][7];

      Real_t hourmody =
          y8n[i3] * gamma[i1][0] + y8n[i3 + 1] * gamma[i1][1] +
          y8n[i3 + 2] * gamma[i1][2] + y8n[i3 + 3] * gamma[i1][3] +
          y8n[i3 + 4] * gamma[i1][4] + y8n[i3 + 5] * gamma[i1][5] +
          y8n[i3 + 6] * gamma[i1][6] + y8n[i3 + 7] * gamma[i1][7];

      Real_t hourmodz =
          z8n[i3] * gamma[i1][0] + z8n[i3 + 1] * gamma[i1][1] +
          z8n[i3 + 2] * gamma[i1][2] + z8n[i3 + 3] * gamma[i1][3] +
          z8n[i3 + 4] * gamma[i1][4] + z8n[i3 + 5] * gamma[i1][5] +
          z8n[i3 + 6] * gamma[i1][6] + z8n[i3 + 7] * gamma[i1][7];

      hourgam[0][i1] =
          gamma[i1][0] - volinv * (dvdx[i3] * hourmodx + dvdy[i3] * hourmody +
                                   dvdz[i3] * hourmodz);

      hourgam[1][i1] = gamma[i1][1] - volinv * (dvdx[i3 + 1] * hourmodx +
                                                dvdy[i3 + 1] * hourmody +
                                                dvdz[i3 + 1] * hourmodz);

      hourgam[2][i1] = gamma[i1][2] - volinv * (dvdx[i3 + 2] * hourmodx +
                                                dvdy[i3 + 2] * hourmody +
                                                dvdz[i3 + 2] * hourmodz);

      hourgam[3][i1] = gamma[i1][3] - volinv * (dvdx[i3 + 3] * hourmodx +
                                                dvdy[i3 + 3] * hourmody +
                                                dvdz[i3 + 3] * hourmodz);

      hourgam[4][i1] = gamma[i1][4] - volinv * (dvdx[i3 + 4] * hourmodx +
                                                dvdy[i3 + 4] * hourmody +
                                                dvdz[i3 + 4] * hourmodz);

      hourgam[5][i1] = gamma[i1][5] - volinv * (dvdx[i3 + 5] * hourmodx +
                                                dvdy[i3 + 5] * hourmody +
                                                dvdz[i3 + 5] * hourmodz);

      hourgam[6][i1] = gamma[i1][6] - volinv * (dvdx[i3 + 6] * hourmodx +
                                                dvdy[i3 + 6] * hourmody +
                                                dvdz[i3 + 6] * hourmodz);

      hourgam[7][i1] = gamma[i1][7] - volinv * (dvdx[i3 + 7] * hourmodx +
                                                dvdy[i3 + 7] * hourmody +
                                                dvdz[i3 + 7] * hourmodz);
    }

    Real_t ss1 = ss[i2];
    Real_t mass1 = elemMass[i2];
    Real_t volume13 = CBRT(determ[i2]);

    Index_t n0si2 = nodelist[i2][0];
    Index_t n1si2 = nodelist[i2][1];
    Index_t n2si2 = nodelist[i2][2];
    Index_t n3si2 = nodelist[i2][3];
    Index_t n4si2 = nodelist[i2][4];
    Index_t n5si2 = nodelist[i2][5];
    Index_t n6si2 = nodelist[i2][6];
    Index_t n7si2 = nodelist[i2][7];

    xd1[0] = xd[n0si2];
    xd1[1] = xd[n1si2];
    xd1[2] = xd[n2si2];
    xd1[3] = xd[n3si2];
    xd1[4] = xd[n4si2];
    xd1[5] = xd[n5si2];
    xd1[6] = xd[n6si2];
    xd1[7] = xd[n7si2];

    yd1[0] = yd[n0si2];
    yd1[1] = yd[n1si2];
    yd1[2] = yd[n2si2];
    yd1[3] = yd[n3si2];
    yd1[4] = yd[n4si2];
    yd1[5] = yd[n5si2];
    yd1[6] = yd[n6si2];
    yd1[7] = yd[n7si2];

    zd1[0] = zd[n0si2];
    zd1[1] = zd[n1si2];
    zd1[2] = zd[n2si2];
    zd1[3] = zd[n3si2];
    zd1[4] = zd[n4si2];
    zd1[5] = zd[n5si2];
    zd1[6] = zd[n6si2];
    zd1[7] = zd[n7si2];

    coefficient = -hourg * 0.01 * ss1 * mass1 / volume13;

    CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coefficient, hgfx, hgfy,
                             hgfz);

    for (int i = 0; i < 8; i++)
      fx_elem[i2][i] = hgfx[i];
    for (int i = 0; i < 8; i++)
      fy_elem[i2][i] = hgfy[i];
    for (int i = 0; i < 8; i++)
      fz_elem[i2][i] = hgfz[i];
  }

#pragma omp parallel for firstprivate(numNode)
  for (Index_t gnode = 0; gnode < numNode; ++gnode) {
    Index_t count = nodeElemStart[gnode + 1] - nodeElemStart[gnode];
    Index_t *cornerList = (Index_t *)&nodeElemCornerList[nodeElemStart[gnode]];
    Real_t fx_tmp = 0.0;
    Real_t fy_tmp = 0.0;
    Real_t fz_tmp = 0.0;
    for (Index_t i = 0; i < count; ++i) {
      Index_t elem = cornerList[i];
      fx_tmp += fx_elem[elem / 8][elem % 8];
      fy_tmp += fy_elem[elem / 8][elem % 8];
      fz_tmp += fz_elem[elem / 8][elem % 8];
    }
    fx[gnode] += fx_tmp;
    fy[gnode] += fy_tmp;
    fz[gnode] += fz_tmp;
  }
  FreeReal2D(fz_elem, numElem);
  FreeReal2D(fy_elem, numElem);
  FreeReal2D(fx_elem, numElem);
}

//**************************************************
// Calculate hourglass control for elements
//**************************************************

static inline void CalcHourglassControlForElems(
    const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
    const Real_t *yd, const Real_t *zd, Index_t **nodelist, const Real_t *volo,
    const Real_t *v, const Real_t *ss, const Real_t *elemMass,
    const Index_t *nodeElemStart, const Index_t *nodeElemCornerList, Real_t *fx,
    Real_t *fy, Real_t *fz, Real_t determ[], Real_t hgcoef, Index_t numElem,
    Index_t numNode) {
  Index_t numElem8 = numElem * 8;
  Real_t *dvdx = AllocateReal(numElem8);
  Real_t *dvdy = AllocateReal(numElem8);
  Real_t *dvdz = AllocateReal(numElem8);
  Real_t *x8n = AllocateReal(numElem8);
  Real_t *y8n = AllocateReal(numElem8);
  Real_t *z8n = AllocateReal(numElem8);

#pragma omp parallel for firstprivate(numElem)
  for (Index_t i = 0; i < numElem; ++i) {
    Real_t x1[8], y1[8], z1[8];
    Real_t pfx[8], pfy[8], pfz[8];

    CollectNodesToElemNodes(x, y, z, nodelist[i], x1, y1, z1);

    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

    for (Index_t ii = 0; ii < 8; ++ii) {
      Index_t jj = 8 * i + ii;
      dvdx[jj] = pfx[ii];
      dvdy[jj] = pfy[ii];
      dvdz[jj] = pfz[ii];
      x8n[jj] = x1[ii];
      y8n[jj] = y1[ii];
      z8n[jj] = z1[ii];
    }

    determ[i] = volo[i] * v[i];

    // if (v[i] <= 0.0) {
    //   exit(VolumeError);
    // }
  }

  if (hgcoef > 0.) {
    CalcFBHourglassForceForElems(xd, yd, zd, nodelist, ss, elemMass,
                                 nodeElemStart, nodeElemCornerList, fx, fy, fz,
                                 determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                 hgcoef, numElem, numNode);
  }

  free(z8n);
  free(y8n);
  free(x8n);
  free(dvdz);
  free(dvdy);
  free(dvdx);
}

  //**************************************************
  // Calculate volume force for elements
  //**************************************************

static inline void CalcVolumeForceForElems(
    const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
    const Real_t *yd, const Real_t *zd, Index_t **nodelist,
    const Real_t *volo, const Real_t *v, const Real_t *p, const Real_t *q,
    const Real_t *ss, const Real_t *elemMass, const Index_t *nodeElemStart,
    const Index_t *nodeElemCornerList, Real_t *fx, Real_t *fy, Real_t *fz,
    Real_t hgcoef, Index_t numElem, Index_t numNode) {
  if (numElem != 0) {
    Real_t *sigxx = AllocateReal(numElem);
    Real_t *sigyy = AllocateReal(numElem);
    Real_t *sigzz = AllocateReal(numElem);
    Real_t *determ = AllocateReal(numElem);

    InitStressTermsForElems(p, q, sigxx, sigyy, sigzz, numElem);

    IntegrateStressForElems(x, y, z, nodelist, nodeElemStart,
                            nodeElemCornerList, fx, fy, fz, sigxx, sigyy,
                            sigzz, determ, numElem, numNode);

    // #pragma omp parallel for firstprivate(numElem)
    // for (Index_t k = 0; k < numElem; ++k) {
    //   if (determ[k] <= 0.0) {
    //     exit(VolumeError);
    //   }
    // }

    CalcHourglassControlForElems(
        x, y, z, xd, yd, zd, nodelist, volo, v, ss, elemMass, nodeElemStart,
        nodeElemCornerList, fx, fy, fz, determ, hgcoef, numElem, numNode);

    free(determ);
    free(sigzz);
    free(sigyy);
    free(sigxx);
  }
}

  //**************************************************
  // Calculate force for nodes
  //**************************************************

static inline void CalcForceForNodes(
    const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
    const Real_t *yd, const Real_t *zd, Index_t **nodelist,
    const Real_t *volo, const Real_t *v, const Real_t *p, const Real_t *q,
    const Real_t *ss, const Real_t *elemMass, const Index_t *nodeElemStart,
    const Index_t *nodeElemCornerList, Real_t *fx, Real_t *fy, Real_t *fz,
    Real_t hgcoef, Index_t numElem, Index_t numNode) {
#pragma omp parallel for firstprivate(numNode)
    for (Index_t i = 0; i < numNode; ++i) {
      fx[i] = 0.0;
      fy[i] = 0.0;
      fz[i] = 0.0;
    }

    CalcVolumeForceForElems(x, y, z, xd, yd, zd, nodelist, volo, v, p, q, ss,
                            elemMass, nodeElemStart, nodeElemCornerList, fx, fy,
                            fz, hgcoef, numElem, numNode);
  }

  //**************************************************
  // Calculate acceleration for nodes
  //**************************************************

static inline void CalcAccelerationForNodes(
    const Real_t *fx, const Real_t *fy, const Real_t *fz,
    const Real_t *nodalMass, Real_t *xdd, Real_t *ydd, Real_t *zdd,
    Index_t numNode) {
#pragma omp parallel for firstprivate(numNode)
    for (Index_t i = 0; i < numNode; ++i) {
      xdd[i] = fx[i] / nodalMass[i];
      ydd[i] = fy[i] / nodalMass[i];
      zdd[i] = fz[i] / nodalMass[i];
    }
  }

  //**************************************************
  // Apply acceleration boundary conditions
  //**************************************************

  static inline void ApplyAccelerationBoundaryConditionsForNodes(
      Real_t * xdd, Real_t * ydd, Real_t * zdd, const Index_t *symmX,
      const Index_t *symmY, const Index_t *symmZ, Index_t symmX_size,
      Index_t symmY_size, Index_t symmZ_size, Index_t sizeX) {
    Index_t numNodeBC = (sizeX + 1) * (sizeX + 1);

    // Simplified for CARTS: use separate parallel loops
    if (symmX_size != 0) {
#pragma omp parallel for firstprivate(numNodeBC)
      for (Index_t i = 0; i < numNodeBC; ++i)
        xdd[symmX[i]] = 0.0;
    }

    if (symmY_size != 0) {
#pragma omp parallel for firstprivate(numNodeBC)
      for (Index_t i = 0; i < numNodeBC; ++i)
        ydd[symmY[i]] = 0.0;
    }

    if (symmZ_size != 0) {
#pragma omp parallel for firstprivate(numNodeBC)
      for (Index_t i = 0; i < numNodeBC; ++i)
        zdd[symmZ[i]] = 0.0;
    }
  }

  //**************************************************
  // Calculate velocity for nodes
  //**************************************************

  static inline void CalcVelocityForNodes(Real_t * xd, Real_t * yd, Real_t * zd,
                                          const Real_t *xdd, const Real_t *ydd,
                                          const Real_t *zdd, const Real_t dt,
                                          const Real_t u_cut, Index_t numNode) {
#pragma omp parallel for firstprivate(numNode)
    for (Index_t i = 0; i < numNode; ++i) {
      Real_t xdtmp = xd[i] + xdd[i] * dt;
      if (FABS(xdtmp) < u_cut)
        xdtmp = 0.0;
      xd[i] = xdtmp;

      Real_t ydtmp = yd[i] + ydd[i] * dt;
      if (FABS(ydtmp) < u_cut)
        ydtmp = 0.0;
      yd[i] = ydtmp;

      Real_t zdtmp = zd[i] + zdd[i] * dt;
      if (FABS(zdtmp) < u_cut)
        zdtmp = 0.0;
      zd[i] = zdtmp;
    }
  }

  //**************************************************
  // Calculate position for nodes
  //**************************************************

  static inline void CalcPositionForNodes(
      Real_t * x, Real_t * y, Real_t * z, const Real_t *xd, const Real_t *yd,
      const Real_t *zd, const Real_t dt, Index_t numNode) {
#pragma omp parallel for firstprivate(numNode)
    for (Index_t i = 0; i < numNode; ++i) {
      x[i] += xd[i] * dt;
      y[i] += yd[i] * dt;
      z[i] += zd[i] * dt;
    }
  }

  //**************************************************
  // Lagrange nodal
  //**************************************************

  static inline void LagrangeNodal(
      Real_t * x, Real_t * y, Real_t * z, Real_t * xd, Real_t * yd, Real_t * zd,
      Real_t * xdd, Real_t * ydd, Real_t * zdd, Real_t * fx, Real_t * fy,
      Real_t * fz, const Real_t *nodalMass, Index_t **nodelist,
      const Real_t *volo, const Real_t *v, const Real_t *p, const Real_t *q,
      const Real_t *ss, const Real_t *elemMass, const Index_t *nodeElemStart,
      const Index_t *nodeElemCornerList, const Index_t *symmX,
      const Index_t *symmY, const Index_t *symmZ, Index_t symmX_size,
      Index_t symmY_size, Index_t symmZ_size, Real_t hgcoef, Real_t deltatime,
      Real_t u_cut, Index_t numElem, Index_t numNode, Index_t sizeX) {
    CalcForceForNodes(x, y, z, xd, yd, zd, nodelist, volo, v, p, q, ss,
                      elemMass, nodeElemStart, nodeElemCornerList, fx, fy, fz,
                      hgcoef, numElem, numNode);
    CalcAccelerationForNodes(fx, fy, fz, nodalMass, xdd, ydd, zdd, numNode);
    ApplyAccelerationBoundaryConditionsForNodes(xdd, ydd, zdd, symmX, symmY,
                                                symmZ, symmX_size, symmY_size,
                                                symmZ_size, sizeX);
    CalcVelocityForNodes(xd, yd, zd, xdd, ydd, zdd, deltatime, u_cut, numNode);
    CalcPositionForNodes(x, y, z, xd, yd, zd, deltatime, numNode);
  }

  //**************************************************
  // Calculate element volume
  //**************************************************

  static inline Real_t CalcElemVolume_Helper(
      const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
      const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7,
      const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
      const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7,
      const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
      const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7) {
    Real_t twelveth = 1.0 / 12.0;

    Real_t dx61 = x6 - x1;
    Real_t dy61 = y6 - y1;
    Real_t dz61 = z6 - z1;

    Real_t dx70 = x7 - x0;
    Real_t dy70 = y7 - y0;
    Real_t dz70 = z7 - z0;

    Real_t dx63 = x6 - x3;
    Real_t dy63 = y6 - y3;
    Real_t dz63 = z6 - z3;

    Real_t dx20 = x2 - x0;
    Real_t dy20 = y2 - y0;
    Real_t dz20 = z2 - z0;

    Real_t dx50 = x5 - x0;
    Real_t dy50 = y5 - y0;
    Real_t dz50 = z5 - z0;

    Real_t dx64 = x6 - x4;
    Real_t dy64 = y6 - y4;
    Real_t dz64 = z6 - z4;

    Real_t dx31 = x3 - x1;
    Real_t dy31 = y3 - y1;
    Real_t dz31 = z3 - z1;

    Real_t dx72 = x7 - x2;
    Real_t dy72 = y7 - y2;
    Real_t dz72 = z7 - z2;

    Real_t dx43 = x4 - x3;
    Real_t dy43 = y4 - y3;
    Real_t dz43 = z4 - z3;

    Real_t dx57 = x5 - x7;
    Real_t dy57 = y5 - y7;
    Real_t dz57 = z5 - z7;

    Real_t dx14 = x1 - x4;
    Real_t dy14 = y1 - y4;
    Real_t dz14 = z1 - z4;

    Real_t dx25 = x2 - x5;
    Real_t dy25 = y2 - y5;
    Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3)                     \
  ((x1) * ((y2) * (z3) - (z2) * (y3)) + (x2) * ((z1) * (y3) - (y1) * (z3)) +   \
   (x3) * ((y1) * (z2) - (z1) * (y2)))

    Real_t volume = TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20, dy31 + dy72, dy63,
                                   dy20, dz31 + dz72, dz63, dz20) +
                    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70, dy43 + dy57, dy64,
                                   dy70, dz43 + dz57, dz64, dz70) +
                    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50, dy14 + dy25, dy61,
                                   dy50, dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

    volume *= twelveth;
    return volume;
  }

  Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8],
                        const Real_t z[8]) {
    return CalcElemVolume_Helper(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], y[0], y[1], y[2], y[3],
        y[4], y[5], y[6], y[7], z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
  }

  //**************************************************
  // Area face
  //**************************************************

  static inline Real_t AreaFace(
      const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
      const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
      const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3) {
    Real_t fx = (x2 - x0) - (x3 - x1);
    Real_t fy = (y2 - y0) - (y3 - y1);
    Real_t fz = (z2 - z0) - (z3 - z1);
    Real_t gx = (x2 - x0) + (x3 - x1);
    Real_t gy = (y2 - y0) + (y3 - y1);
    Real_t gz = (z2 - z0) + (z3 - z1);
    Real_t area =
        (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) * (fx * gx + fy * gy + fz * gz);
    return area;
  }

  //**************************************************
  // Calculate element characteristic length
  //**************************************************

  static inline Real_t CalcElemCharacteristicLength(
      const Real_t x[8], const Real_t y[8], const Real_t z[8],
      const Real_t volume) {
    Real_t a, charLength = 0.0;

    a = AreaFace(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1],
                 z[2], z[3]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7], z[4], z[5],
                 z[6], z[7]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[0], x[1], x[5], x[4], y[0], y[1], y[5], y[4], z[0], z[1],
                 z[5], z[4]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[1], x[2], x[6], x[5], y[1], y[2], y[6], y[5], z[1], z[2],
                 z[6], z[5]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[2], x[3], x[7], x[6], y[2], y[3], y[7], y[6], z[2], z[3],
                 z[7], z[6]);
    charLength = MAX(a, charLength);

    a = AreaFace(x[3], x[0], x[4], x[7], y[3], y[0], y[4], y[7], z[3], z[0],
                 z[4], z[7]);
    charLength = MAX(a, charLength);

    charLength = 4.0 * volume / SQRT(charLength);
    return charLength;
  }

  //**************************************************
  // Calculate element velocity gradient
  //**************************************************

  static inline void CalcElemVelocityGradient(
      const Real_t *const xvel, const Real_t *const yvel,
      const Real_t *const zvel, const Real_t b[][8], const Real_t detJ,
      Real_t *const d) {
    const Real_t inv_detJ = 1.0 / detJ;
    const Real_t *const pfx = b[0];
    const Real_t *const pfy = b[1];
    const Real_t *const pfz = b[2];

    d[0] = inv_detJ *
           (pfx[0] * (xvel[0] - xvel[6]) + pfx[1] * (xvel[1] - xvel[7]) +
            pfx[2] * (xvel[2] - xvel[4]) + pfx[3] * (xvel[3] - xvel[5]));

    d[1] = inv_detJ *
           (pfy[0] * (yvel[0] - yvel[6]) + pfy[1] * (yvel[1] - yvel[7]) +
            pfy[2] * (yvel[2] - yvel[4]) + pfy[3] * (yvel[3] - yvel[5]));

    d[2] = inv_detJ *
           (pfz[0] * (zvel[0] - zvel[6]) + pfz[1] * (zvel[1] - zvel[7]) +
            pfz[2] * (zvel[2] - zvel[4]) + pfz[3] * (zvel[3] - zvel[5]));

    Real_t dyddx =
        inv_detJ *
        (pfx[0] * (yvel[0] - yvel[6]) + pfx[1] * (yvel[1] - yvel[7]) +
         pfx[2] * (yvel[2] - yvel[4]) + pfx[3] * (yvel[3] - yvel[5]));

    Real_t dxddy =
        inv_detJ *
        (pfy[0] * (xvel[0] - xvel[6]) + pfy[1] * (xvel[1] - xvel[7]) +
         pfy[2] * (xvel[2] - xvel[4]) + pfy[3] * (xvel[3] - xvel[5]));

    Real_t dzddx =
        inv_detJ *
        (pfx[0] * (zvel[0] - zvel[6]) + pfx[1] * (zvel[1] - zvel[7]) +
         pfx[2] * (zvel[2] - zvel[4]) + pfx[3] * (zvel[3] - zvel[5]));

    Real_t dxddz =
        inv_detJ *
        (pfz[0] * (xvel[0] - xvel[6]) + pfz[1] * (xvel[1] - xvel[7]) +
         pfz[2] * (xvel[2] - xvel[4]) + pfz[3] * (xvel[3] - xvel[5]));

    Real_t dzddy =
        inv_detJ *
        (pfy[0] * (zvel[0] - zvel[6]) + pfy[1] * (zvel[1] - zvel[7]) +
         pfy[2] * (zvel[2] - zvel[4]) + pfy[3] * (zvel[3] - zvel[5]));

    Real_t dyddz =
        inv_detJ *
        (pfz[0] * (yvel[0] - yvel[6]) + pfz[1] * (yvel[1] - yvel[7]) +
         pfz[2] * (yvel[2] - yvel[4]) + pfz[3] * (yvel[3] - yvel[5]));

    d[5] = 0.5 * (dxddy + dyddx);
    d[4] = 0.5 * (dxddz + dzddx);
    d[3] = 0.5 * (dzddy + dyddz);
  }

  //**************************************************
  // Calculate kinematics for elements
  //**************************************************

  static inline void CalcKinematicsForElems(
      const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
      const Real_t *yd, const Real_t *zd, const Real_t *volo, const Real_t *v,
      Real_t *delv, Real_t *arealg, Real_t *dxx, Real_t *dyy, Real_t *dzz,
      Index_t **nodelist, Real_t *vnew, Real_t deltaTime, Index_t numElem) {
#pragma omp parallel for firstprivate(numElem)
    for (Index_t k = 0; k < numElem; ++k) {
      Real_t B[3][8];
      Real_t D[6];
      Real_t x_local[8];
      Real_t y_local[8];
      Real_t z_local[8];
      Real_t xd_local[8];
      Real_t yd_local[8];
      Real_t zd_local[8];
      Real_t detJ = 0.0;

      Real_t volume;
      Real_t relativeVolume;

      CollectNodesToElemNodes(x, y, z, nodelist[k], x_local, y_local, z_local);

      volume = CalcElemVolume(x_local, y_local, z_local);
      relativeVolume = volume / volo[k];
      vnew[k] = relativeVolume;
      delv[k] = relativeVolume - v[k];

      arealg[k] =
          CalcElemCharacteristicLength(x_local, y_local, z_local, volume);

      for (Index_t lnode = 0; lnode < 8; ++lnode) {
        Index_t gnode = nodelist[k][lnode];
        xd_local[lnode] = xd[gnode];
        yd_local[lnode] = yd[gnode];
        zd_local[lnode] = zd[gnode];
      }

      Real_t dt2 = 0.5 * deltaTime;
      for (Index_t j = 0; j < 8; ++j) {
        x_local[j] -= dt2 * xd_local[j];
        y_local[j] -= dt2 * yd_local[j];
        z_local[j] -= dt2 * zd_local[j];
      }

      CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);

      CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D);

      dxx[k] = D[0];
      dyy[k] = D[1];
      dzz[k] = D[2];
    }
  }

  //**************************************************
  // Calculate Lagrange elements
  //**************************************************

  static inline void CalcLagrangeElements(
      const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
      const Real_t *yd, const Real_t *zd, const Real_t *volo, const Real_t *v,
      Real_t *delv, Real_t *arealg, Real_t *vdov, Index_t **nodelist,
      Real_t *vnew, Real_t deltatime, Index_t numElem) {
    if (numElem > 0) {
      // Allocate temporary strain arrays
      Real_t *dxx = AllocateReal(numElem);
      Real_t *dyy = AllocateReal(numElem);
      Real_t *dzz = AllocateReal(numElem);

      CalcKinematicsForElems(x, y, z, xd, yd, zd, volo, v, delv, arealg, dxx,
                             dyy, dzz, nodelist, vnew, deltatime, numElem);

#pragma omp parallel for firstprivate(numElem)
      for (Index_t k = 0; k < numElem; ++k) {
        Real_t vdov_k = dxx[k] + dyy[k] + dzz[k];
        Real_t vdovthird = vdov_k / 3.0;

        vdov[k] = vdov_k;
        dxx[k] -= vdovthird;
        dyy[k] -= vdovthird;
        dzz[k] -= vdovthird;
        // if (vnew[k] <= 0.0) {
        //   exit(VolumeError);
        // }
      }

      // Deallocate temporary strain arrays
      free(dxx);
      free(dyy);
      free(dzz);
    }
  }

  //**************************************************
  // Calculate monotonic Q gradients for elements
  //**************************************************

  static inline void CalcMonotonicQGradientsForElems(
      const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
      const Real_t *yd, const Real_t *zd, const Real_t *volo,
      Index_t **nodelist, Real_t *delx_zeta, Real_t *delv_zeta, Real_t *delx_xi,
      Real_t *delv_xi, Real_t *delx_eta, Real_t *delv_eta, Real_t vnew[],
      Index_t numElem) {
    const Real_t ptiny = 1.e-36;

#pragma omp parallel for firstprivate(numElem)
    for (Index_t i = 0; i < numElem; ++i) {
      Real_t ax, ay, az;
      Real_t dxv, dyv, dzv;

      Index_t n0 = nodelist[i][0];
      Index_t n1 = nodelist[i][1];
      Index_t n2 = nodelist[i][2];
      Index_t n3 = nodelist[i][3];
      Index_t n4 = nodelist[i][4];
      Index_t n5 = nodelist[i][5];
      Index_t n6 = nodelist[i][6];
      Index_t n7 = nodelist[i][7];

      Real_t x0 = x[n0];
      Real_t x1 = x[n1];
      Real_t x2 = x[n2];
      Real_t x3 = x[n3];
      Real_t x4 = x[n4];
      Real_t x5 = x[n5];
      Real_t x6 = x[n6];
      Real_t x7 = x[n7];

      Real_t y0 = y[n0];
      Real_t y1 = y[n1];
      Real_t y2 = y[n2];
      Real_t y3 = y[n3];
      Real_t y4 = y[n4];
      Real_t y5 = y[n5];
      Real_t y6 = y[n6];
      Real_t y7 = y[n7];

      Real_t z0 = z[n0];
      Real_t z1 = z[n1];
      Real_t z2 = z[n2];
      Real_t z3 = z[n3];
      Real_t z4 = z[n4];
      Real_t z5 = z[n5];
      Real_t z6 = z[n6];
      Real_t z7 = z[n7];

      Real_t xv0 = xd[n0];
      Real_t xv1 = xd[n1];
      Real_t xv2 = xd[n2];
      Real_t xv3 = xd[n3];
      Real_t xv4 = xd[n4];
      Real_t xv5 = xd[n5];
      Real_t xv6 = xd[n6];
      Real_t xv7 = xd[n7];

      Real_t yv0 = yd[n0];
      Real_t yv1 = yd[n1];
      Real_t yv2 = yd[n2];
      Real_t yv3 = yd[n3];
      Real_t yv4 = yd[n4];
      Real_t yv5 = yd[n5];
      Real_t yv6 = yd[n6];
      Real_t yv7 = yd[n7];

      Real_t zv0 = zd[n0];
      Real_t zv1 = zd[n1];
      Real_t zv2 = zd[n2];
      Real_t zv3 = zd[n3];
      Real_t zv4 = zd[n4];
      Real_t zv5 = zd[n5];
      Real_t zv6 = zd[n6];
      Real_t zv7 = zd[n7];

      Real_t vol = volo[i] * vnew[i];
      Real_t norm = 1.0 / (vol + ptiny);

      Real_t dxj = -0.25 * ((x0 + x1 + x5 + x4) - (x3 + x2 + x6 + x7));
      Real_t dyj = -0.25 * ((y0 + y1 + y5 + y4) - (y3 + y2 + y6 + y7));
      Real_t dzj = -0.25 * ((z0 + z1 + z5 + z4) - (z3 + z2 + z6 + z7));

      Real_t dxi = 0.25 * ((x1 + x2 + x6 + x5) - (x0 + x3 + x7 + x4));
      Real_t dyi = 0.25 * ((y1 + y2 + y6 + y5) - (y0 + y3 + y7 + y4));
      Real_t dzi = 0.25 * ((z1 + z2 + z6 + z5) - (z0 + z3 + z7 + z4));

      Real_t dxk = 0.25 * ((x4 + x5 + x6 + x7) - (x0 + x1 + x2 + x3));
      Real_t dyk = 0.25 * ((y4 + y5 + y6 + y7) - (y0 + y1 + y2 + y3));
      Real_t dzk = 0.25 * ((z4 + z5 + z6 + z7) - (z0 + z1 + z2 + z3));

      /* find delvk and delxk ( i cross j ) */
      ax = dyi * dzj - dzi * dyj;
      ay = dzi * dxj - dxi * dzj;
      az = dxi * dyj - dyi * dxj;

      delx_zeta[i] = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

      ax *= norm;
      ay *= norm;
      az *= norm;

      dxv = 0.25 * ((xv4 + xv5 + xv6 + xv7) - (xv0 + xv1 + xv2 + xv3));
      dyv = 0.25 * ((yv4 + yv5 + yv6 + yv7) - (yv0 + yv1 + yv2 + yv3));
      dzv = 0.25 * ((zv4 + zv5 + zv6 + zv7) - (zv0 + zv1 + zv2 + zv3));

      delv_zeta[i] = ax * dxv + ay * dyv + az * dzv;

      /* find delxi and delvi ( j cross k ) */
      ax = dyj * dzk - dzj * dyk;
      ay = dzj * dxk - dxj * dzk;
      az = dxj * dyk - dyj * dxk;

      delx_xi[i] = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

      ax *= norm;
      ay *= norm;
      az *= norm;

      dxv = 0.25 * ((xv1 + xv2 + xv6 + xv5) - (xv0 + xv3 + xv7 + xv4));
      dyv = 0.25 * ((yv1 + yv2 + yv6 + yv5) - (yv0 + yv3 + yv7 + yv4));
      dzv = 0.25 * ((zv1 + zv2 + zv6 + zv5) - (zv0 + zv3 + zv7 + zv4));

      delv_xi[i] = ax * dxv + ay * dyv + az * dzv;

      /* find delxj and delvj ( k cross i ) */
      ax = dyk * dzi - dzk * dyi;
      ay = dzk * dxi - dxk * dzi;
      az = dxk * dyi - dyk * dxi;

      delx_eta[i] = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);

      ax *= norm;
      ay *= norm;
      az *= norm;

      dxv = -0.25 * ((xv0 + xv1 + xv5 + xv4) - (xv3 + xv2 + xv6 + xv7));
      dyv = -0.25 * ((yv0 + yv1 + yv5 + yv4) - (yv3 + yv2 + yv6 + yv7));
      dzv = -0.25 * ((zv0 + zv1 + zv5 + zv4) - (zv3 + zv2 + zv6 + zv7));

      delv_eta[i] = ax * dxv + ay * dyv + az * dzv;
    }
  }

  //**************************************************
  // Calculate monotonic Q region for elements
  //**************************************************

  static inline void CalcMonotonicQRegionForElems(
      const Real_t *delx_zeta, const Real_t *delv_zeta, const Real_t *delx_xi,
      const Real_t *delv_xi, const Real_t *delx_eta, const Real_t *delv_eta,
      const Real_t *vdov, const Real_t *elemMass, const Real_t *volo,
      const Int_t *elemBC, const Index_t *lxim, const Index_t *lxip,
      const Index_t *letam, const Index_t *letap, const Index_t *lzetam,
      const Index_t *lzetap, Index_t regElemSize, Real_t *qq, Real_t *ql,
      Real_t vnew[], Real_t ptiny, Real_t monoq_limiter_mult,
      Real_t monoq_max_slope, Real_t qlc_monoq, Real_t qqc_monoq) {

    for (Index_t i = 0; i < regElemSize; ++i) {
      Real_t qlin, qquad;
      Real_t phixi, phieta, phizeta;
      Int_t bcMask = elemBC[i];
      Real_t delvm = 0.0, delvp = 0.0;

      /*  phixi     */
      Real_t norm = 1.0 / (delv_xi[i] + ptiny);

      // Note: Using if-else instead of switch due to polygeist fallthrough bug
      {
        Int_t xiMask = bcMask & XI_M;
        if (xiMask == XI_M_COMM || xiMask == 0) {
          delvm = delv_xi[lxim[i]];
        } else if (xiMask == XI_M_SYMM) {
          delvm = delv_xi[i];
        } else if (xiMask == XI_M_FREE) {
          delvm = 0.0;
        } else {
          fprintf(stderr, "Error in switch at %s line %d\n", __FILE__,
                  __LINE__);
          delvm = 0;
        }
      }
      {
        Int_t xiPMask = bcMask & XI_P;
        if (xiPMask == XI_P_COMM || xiPMask == 0) {
          delvp = delv_xi[lxip[i]];
        } else if (xiPMask == XI_P_SYMM) {
          delvp = delv_xi[i];
        } else if (xiPMask == XI_P_FREE) {
          delvp = 0.0;
        } else {
          fprintf(stderr, "Error in switch at %s line %d\n", __FILE__,
                  __LINE__);
          delvp = 0;
        }
      }

      delvm = delvm * norm;
      delvp = delvp * norm;

      phixi = 0.5 * (delvm + delvp);

      delvm *= monoq_limiter_mult;
      delvp *= monoq_limiter_mult;

      if (delvm < phixi)
        phixi = delvm;
      if (delvp < phixi)
        phixi = delvp;
      if (phixi < 0.0)
        phixi = 0.0;
      if (phixi > monoq_max_slope)
        phixi = monoq_max_slope;

      /*  phieta     */
      norm = 1.0 / (delv_eta[i] + ptiny);

      // Note: Using if-else instead of switch due to polygeist fallthrough bug
      {
        Int_t etaMask = bcMask & ETA_M;
        if (etaMask == ETA_M_COMM || etaMask == 0) {
          delvm = delv_eta[letam[i]];
        } else if (etaMask == ETA_M_SYMM) {
          delvm = delv_eta[i];
        } else if (etaMask == ETA_M_FREE) {
          delvm = 0.0;
        } else {
          fprintf(stderr, "Error in switch at %s line %d\n", __FILE__,
                  __LINE__);
          delvm = 0;
        }
      }
      {
        Int_t etaPMask = bcMask & ETA_P;
        if (etaPMask == ETA_P_COMM || etaPMask == 0) {
          delvp = delv_eta[letap[i]];
        } else if (etaPMask == ETA_P_SYMM) {
          delvp = delv_eta[i];
        } else if (etaPMask == ETA_P_FREE) {
          delvp = 0.0;
        } else {
          fprintf(stderr, "Error in switch at %s line %d\n", __FILE__,
                  __LINE__);
          delvp = 0;
        }
      }

      delvm = delvm * norm;
      delvp = delvp * norm;

      phieta = 0.5 * (delvm + delvp);

      delvm *= monoq_limiter_mult;
      delvp *= monoq_limiter_mult;

      if (delvm < phieta)
        phieta = delvm;
      if (delvp < phieta)
        phieta = delvp;
      if (phieta < 0.0)
        phieta = 0.0;
      if (phieta > monoq_max_slope)
        phieta = monoq_max_slope;

      /*  phizeta     */
      norm = 1.0 / (delv_zeta[i] + ptiny);

      // Note: Using if-else instead of switch due to polygeist fallthrough bug
      {
        Int_t zetaMask = bcMask & ZETA_M;
        if (zetaMask == ZETA_M_COMM || zetaMask == 0) {
          delvm = delv_zeta[lzetam[i]];
        } else if (zetaMask == ZETA_M_SYMM) {
          delvm = delv_zeta[i];
        } else if (zetaMask == ZETA_M_FREE) {
          delvm = 0.0;
        } else {
          fprintf(stderr, "Error in switch at %s line %d\n", __FILE__,
                  __LINE__);
          delvm = 0;
        }
      }
      {
        Int_t zetaPMask = bcMask & ZETA_P;
        if (zetaPMask == ZETA_P_COMM || zetaPMask == 0) {
          delvp = delv_zeta[lzetap[i]];
        } else if (zetaPMask == ZETA_P_SYMM) {
          delvp = delv_zeta[i];
        } else if (zetaPMask == ZETA_P_FREE) {
          delvp = 0.0;
        } else {
          fprintf(stderr, "Error in switch at %s line %d\n", __FILE__,
                  __LINE__);
          delvp = 0;
        }
      }

      delvm = delvm * norm;
      delvp = delvp * norm;

      phizeta = 0.5 * (delvm + delvp);

      delvm *= monoq_limiter_mult;
      delvp *= monoq_limiter_mult;

      if (delvm < phizeta)
        phizeta = delvm;
      if (delvp < phizeta)
        phizeta = delvp;
      if (phizeta < 0.0)
        phizeta = 0.0;
      if (phizeta > monoq_max_slope)
        phizeta = monoq_max_slope;

      /* Remove length scale */
      if (vdov[i] > 0.0) {
        qlin = 0.0;
        qquad = 0.0;
      } else {
        Real_t delvxxi = delv_xi[i] * delx_xi[i];
        Real_t delvxeta = delv_eta[i] * delx_eta[i];
        Real_t delvxzeta = delv_zeta[i] * delx_zeta[i];

        if (delvxxi > 0.0)
          delvxxi = 0.0;
        if (delvxeta > 0.0)
          delvxeta = 0.0;
        if (delvxzeta > 0.0)
          delvxzeta = 0.0;

        Real_t rho = elemMass[i] / (volo[i] * vnew[i]);

        qlin = -qlc_monoq * rho *
               (delvxxi * (1.0 - phixi) + delvxeta * (1.0 - phieta) +
                delvxzeta * (1.0 - phizeta));

        qquad = qqc_monoq * rho *
                (delvxxi * delvxxi * (1.0 - phixi * phixi) +
                 delvxeta * delvxeta * (1.0 - phieta * phieta) +
                 delvxzeta * delvxzeta * (1.0 - phizeta * phizeta));
      }
      qq[i] = qquad;
      ql[i] = qlin;
    }
  }

  //**************************************************
  // Calculate monotonic Q for elements
  //**************************************************

  static inline void CalcMonotonicQForElems(
      const Real_t *delx_zeta, const Real_t *delv_zeta, const Real_t *delx_xi,
      const Real_t *delv_xi, const Real_t *delx_eta, const Real_t *delv_eta,
      const Real_t *vdov, const Real_t *elemMass, const Real_t *volo,
      const Int_t *elemBC, const Index_t *lxim, const Index_t *lxip,
      const Index_t *letam, const Index_t *letap, const Index_t *lzetam,
      const Index_t *lzetap, const Index_t *regElemSize, Real_t *qq, Real_t *ql,
      Real_t vnew[], Index_t numReg, Real_t monoq_limiter_mult,
      Real_t monoq_max_slope, Real_t qlc_monoq, Real_t qqc_monoq) {
    const Real_t ptiny = 1.e-36;

    for (Index_t r = numReg - 1; r >= 0; --r) {
      if (regElemSize[r] > 0) {
        CalcMonotonicQRegionForElems(
            delx_zeta, delv_zeta, delx_xi, delv_xi, delx_eta, delv_eta, vdov,
            elemMass, volo, elemBC, lxim, lxip, letam, letap, lzetam, lzetap,
            regElemSize[r], qq, ql, vnew, ptiny, monoq_limiter_mult,
            monoq_max_slope, qlc_monoq, qqc_monoq);
      }
    }
  }

  static inline void CalcQForElems(
      const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
      const Real_t *yd, const Real_t *zd, const Real_t *volo,
      const Real_t *vdov, const Real_t *elemMass, Index_t **nodelist,
      const Int_t *elemBC, const Index_t *lxim, const Index_t *lxip,
      const Index_t *letam, const Index_t *letap, const Index_t *lzetam,
      const Index_t *lzetap, const Index_t *regElemSize, Real_t *q, Real_t *qq,
      Real_t *ql, Real_t vnew[], Index_t numElem, Index_t numReg, Index_t sizeX,
      Index_t sizeY, Index_t sizeZ, Real_t qstop, Real_t monoq_limiter_mult,
      Real_t monoq_max_slope, Real_t qlc_monoq, Real_t qqc_monoq) {

    if (numElem != 0) {
      // Allocate gradient arrays
      Real_t *delx_zeta = AllocateReal(numElem);
      Real_t *delv_zeta = AllocateReal(numElem);
      Real_t *delx_xi = AllocateReal(numElem);
      Real_t *delv_xi = AllocateReal(numElem);
      Real_t *delx_eta = AllocateReal(numElem);
      Real_t *delv_eta = AllocateReal(numElem);

      CalcMonotonicQGradientsForElems(x, y, z, xd, yd, zd, volo, nodelist,
                                      delx_zeta, delv_zeta, delx_xi, delv_xi,
                                      delx_eta, delv_eta, vnew, numElem);
      CalcMonotonicQForElems(delx_zeta, delv_zeta, delx_xi, delv_xi, delx_eta,
                             delv_eta, vdov, elemMass, volo, elemBC, lxim, lxip,
                             letam, letap, lzetam, lzetap, regElemSize, qq, ql,
                             vnew, numReg, monoq_limiter_mult, monoq_max_slope,
                             qlc_monoq, qqc_monoq);

      // Deallocate gradient arrays
      free(delx_zeta);
      free(delv_zeta);
      free(delx_xi);
      free(delv_xi);
      free(delx_eta);
      free(delv_eta);

      Index_t idx = -1;
      for (Index_t i = 0; i < numElem; ++i) {
        if (q[i] > qstop) {
          idx = i;
          break;
        }
      }
      // if (idx >= 0) {
      //   exit(QStopError);
      // }
    }
  }

  //**************************************************
  // Simplified EOS functions
  //**************************************************

  static inline void CalcPressureForElems(
      Real_t * p_new, Real_t * bvc, Real_t * pbvc, Real_t * e_old,
      Real_t * compression, Real_t * vnewc, Real_t pmin, Real_t p_cut,
      Real_t eosvmax, Index_t length) {

#pragma omp parallel for firstprivate(length)
    for (Index_t i = 0; i < length; ++i) {
      bvc[i] = 2.0 / 3.0 * (compression[i] + 1.0);
      pbvc[i] = 2.0 / 3.0;
    }

#pragma omp parallel for firstprivate(length, pmin, p_cut, eosvmax)
    for (Index_t i = 0; i < length; ++i) {
      p_new[i] = bvc[i] * e_old[i];

      if (FABS(p_new[i]) < p_cut)
        p_new[i] = 0.0;
      if (vnewc[i] >= eosvmax)
        p_new[i] = 0.0;
      if (p_new[i] < pmin)
        p_new[i] = pmin;
    }
  }

  static inline void CalcEnergyForElems(
      Real_t * p_new, Real_t * e_new, Real_t * q_new, Real_t * bvc,
      Real_t * pbvc, Real_t * p_old, Real_t * e_old, Real_t * q_old,
      Real_t * compression, Real_t * compHalfStep, Real_t * vnewc,
      Real_t * work, Real_t * delvc, Real_t pmin, Real_t p_cut, Real_t e_cut,
      Real_t q_cut, Real_t emin, Real_t * qq_old, Real_t * ql_old, Real_t rho0,
      Real_t eosvmax, Index_t length) {

    Real_t *pHalfStep = AllocateReal(length);

#pragma omp parallel for firstprivate(length, emin)
    for (Index_t i = 0; i < length; ++i) {
      e_new[i] =
          e_old[i] - 0.5 * delvc[i] * (p_old[i] + q_old[i]) + 0.5 * work[i];
      if (e_new[i] < emin) {
        e_new[i] = emin;
      }
    }

    CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc, pmin,
                         p_cut, eosvmax, length);

#pragma omp parallel for firstprivate(length, rho0)
    for (Index_t i = 0; i < length; ++i) {
      Real_t vhalf = 1.0 / (1.0 + compHalfStep[i]);
      Real_t ssc = 0.0;
      if (delvc[i] > 0.0) {
        q_new[i] = 0.0;
      } else {
        ssc =
            (pbvc[i] * e_new[i] + vhalf * vhalf * bvc[i] * pHalfStep[i]) / rho0;
        if (ssc <= 0.1111111e-36) {
          ssc = 0.3333333e-18;
        } else {
          ssc = SQRT(ssc);
        }
        q_new[i] = (ssc * ql_old[i] + qq_old[i]);
      }
      e_new[i] = e_new[i] + 0.5 * delvc[i] *
                                (3.0 * (p_old[i] + q_old[i]) -
                                 4.0 * (pHalfStep[i] + q_new[i]));
    }

#pragma omp parallel for firstprivate(length, emin, e_cut)
    for (Index_t i = 0; i < length; ++i) {
      e_new[i] += 0.5 * work[i];
      if (FABS(e_new[i]) < e_cut) {
        e_new[i] = 0.0;
      }
      if (e_new[i] < emin) {
        e_new[i] = emin;
      }
    }

    CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc, pmin,
                         p_cut, eosvmax, length);

#pragma omp parallel for firstprivate(length, rho0, emin, e_cut)
    for (Index_t i = 0; i < length; ++i) {
      const Real_t sixth = 1.0 / 6.0;
      Real_t q_tilde;

      if (delvc[i] > 0.0) {
        q_tilde = 0.0;
      } else {
        Real_t ssc =
            (pbvc[i] * e_new[i] + vnewc[i] * vnewc[i] * bvc[i] * p_new[i]) /
            rho0;
        if (ssc <= 0.1111111e-36) {
          ssc = 0.3333333e-18;
        } else {
          ssc = SQRT(ssc);
        }
        q_tilde = (ssc * ql_old[i] + qq_old[i]);
      }

      e_new[i] =
          e_new[i] - (7.0 * (p_old[i] + q_old[i]) -
                      8.0 * (pHalfStep[i] + q_new[i]) + (p_new[i] + q_tilde)) *
                         delvc[i] * sixth;

      if (FABS(e_new[i]) < e_cut) {
        e_new[i] = 0.0;
      }
      if (e_new[i] < emin) {
        e_new[i] = emin;
      }
    }

    CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc, pmin,
                         p_cut, eosvmax, length);

#pragma omp parallel for firstprivate(length, rho0, q_cut)
    for (Index_t i = 0; i < length; ++i) {
      if (delvc[i] <= 0.0) {
        Real_t ssc =
            (pbvc[i] * e_new[i] + vnewc[i] * vnewc[i] * bvc[i] * p_new[i]) /
            rho0;
        if (ssc <= 0.1111111e-36) {
          ssc = 0.3333333e-18;
        } else {
          ssc = SQRT(ssc);
        }
        q_new[i] = (ssc * ql_old[i] + qq_old[i]);
        if (FABS(q_new[i]) < q_cut)
          q_new[i] = 0.0;
      }
    }

    free(pHalfStep);
  }

  static inline void CalcSoundSpeedForElems(
      Real_t * ss, Real_t * vnewc, Real_t rho0, Real_t * enewc, Real_t * pnewc,
      Real_t * pbvc, Real_t * bvc, Real_t ss4o3, Index_t len) {

#pragma omp parallel for firstprivate(rho0, ss4o3)
    for (Index_t i = 0; i < len; ++i) {
      Real_t ssTmp =
          (pbvc[i] * enewc[i] + vnewc[i] * vnewc[i] * bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= 0.1111111e-36) {
        ssTmp = 0.3333333e-18;
      } else {
        ssTmp = SQRT(ssTmp);
      }
      ss[i] = ssTmp;
    }
  }

  static inline void EvalEOSForElems(
      Real_t * e, Real_t * p, Real_t * q, Real_t * qq, Real_t * ql, Real_t * ss,
      const Real_t *delv, Real_t *vnewc, Int_t numElemReg, Int_t rep,
      Real_t e_cut, Real_t p_cut, Real_t ss4o3, Real_t q_cut, Real_t eosvmax,
      Real_t eosvmin, Real_t pmin, Real_t emin, Real_t rho0) {

    Real_t *e_old = AllocateReal(numElemReg);
    Real_t *delvc = AllocateReal(numElemReg);
    Real_t *p_old = AllocateReal(numElemReg);
    Real_t *q_old = AllocateReal(numElemReg);
    Real_t *compression = AllocateReal(numElemReg);
    Real_t *compHalfStep = AllocateReal(numElemReg);
    Real_t *qq_old = AllocateReal(numElemReg);
    Real_t *ql_old = AllocateReal(numElemReg);
    Real_t *work = AllocateReal(numElemReg);
    Real_t *p_new = AllocateReal(numElemReg);
    Real_t *e_new = AllocateReal(numElemReg);
    Real_t *q_new = AllocateReal(numElemReg);
    Real_t *bvc = AllocateReal(numElemReg);
    Real_t *pbvc = AllocateReal(numElemReg);

    for (Int_t j = 0; j < rep; j++) {
// Simplified for CARTS: use separate parallel loops
#pragma omp parallel for firstprivate(numElemReg)
      for (Index_t i = 0; i < numElemReg; ++i) {
        e_old[i] = e[i];
        delvc[i] = delv[i];
        p_old[i] = p[i];
        q_old[i] = q[i];
        qq_old[i] = qq[i];
        ql_old[i] = ql[i];
      }

#pragma omp parallel for firstprivate(numElemReg)
      for (Index_t i = 0; i < numElemReg; ++i) {
        Real_t vchalf;
        compression[i] = 1.0 / vnewc[i] - 1.0;
        vchalf = vnewc[i] - delvc[i] * 0.5;
        compHalfStep[i] = 1.0 / vchalf - 1.0;
      }

      if (eosvmin != 0.0) {
#pragma omp parallel for firstprivate(numElemReg, eosvmin)
        for (Index_t i = 0; i < numElemReg; ++i) {
          if (vnewc[i] <= eosvmin) {
            compHalfStep[i] = compression[i];
          }
        }
      }

      if (eosvmax != 0.0) {
#pragma omp parallel for firstprivate(numElemReg, eosvmax)
        for (Index_t i = 0; i < numElemReg; ++i) {
          if (vnewc[i] >= eosvmax) {
            p_old[i] = 0.0;
            compression[i] = 0.0;
            compHalfStep[i] = 0.0;
          }
        }
      }

#pragma omp parallel for firstprivate(numElemReg)
      for (Index_t i = 0; i < numElemReg; ++i) {
        work[i] = 0.0;
      }

      CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc, p_old, e_old, q_old,
                         compression, compHalfStep, vnewc, work, delvc, pmin,
                         p_cut, e_cut, q_cut, emin, qq_old, ql_old, rho0,
                         eosvmax, numElemReg);
    }

#pragma omp parallel for firstprivate(numElemReg)
    for (Index_t i = 0; i < numElemReg; ++i) {
      p[i] = p_new[i];
      e[i] = e_new[i];
      q[i] = q_new[i];
    }

    CalcSoundSpeedForElems(ss, vnewc, rho0, e_new, p_new, pbvc, bvc, ss4o3,
                           numElemReg);

    free(pbvc);
    free(bvc);
    free(q_new);
    free(e_new);
    free(p_new);
    free(work);
    free(ql_old);
    free(qq_old);
    free(compHalfStep);
    free(compression);
    free(q_old);
    free(p_old);
    free(delvc);
    free(e_old);
  }

  static inline void ApplyMaterialPropertiesForElems(
      Real_t * e, Real_t * p, Real_t * q, Real_t * qq, Real_t * ql, Real_t * ss,
      const Real_t *v, const Real_t *delv, Real_t vnew[],
      const Index_t *regElemSize, Index_t numElem, Index_t numReg, Int_t cost,
      Real_t eosvmin, Real_t eosvmax, Real_t e_cut, Real_t p_cut, Real_t ss4o3,
      Real_t q_cut, Real_t pmin, Real_t emin, Real_t rho0) {

    if (numElem != 0) {
      // Simplified for CARTS: use separate parallel loops
      if (eosvmin != 0.0) {
#pragma omp parallel for firstprivate(numElem)
        for (Index_t i = 0; i < numElem; ++i) {
          if (vnew[i] < eosvmin)
            vnew[i] = eosvmin;
        }
      }

      if (eosvmax != 0.0) {
#pragma omp parallel for firstprivate(numElem)
        for (Index_t i = 0; i < numElem; ++i) {
          if (vnew[i] > eosvmax)
            vnew[i] = eosvmax;
        }
      }

#pragma omp parallel for firstprivate(numElem)
      for (Index_t i = 0; i < numElem; ++i) {
        Real_t vc = v[i];
        if (eosvmin != 0.0) {
          if (vc < eosvmin)
            vc = eosvmin;
        }
        if (eosvmax != 0.0) {
          if (vc > eosvmax)
            vc = eosvmax;
        }
        // if (vc <= 0.) {
        //   exit(VolumeError);
        // }
      }

      for (Int_t r = 0; r < numReg; r++) {
        Index_t numElemReg = regElemSize[r];
        Int_t rep;
        if (r < numReg / 2)
          rep = 1;
        else if (r < (numReg - (numReg + 15) / 20))
          rep = 1 + cost;
        else
          rep = 10 * (1 + cost);
        EvalEOSForElems(e, p, q, qq, ql, ss, delv, vnew, numElemReg, rep, e_cut,
                        p_cut, ss4o3, q_cut, eosvmax, eosvmin, pmin, emin,
                        rho0);
      }
    }
  }

  static inline void UpdateVolumesForElems(Real_t * v, Real_t * vnew,
                                           Real_t v_cut, Index_t length) {
    if (length != 0) {
#pragma omp parallel for firstprivate(length, v_cut)
      for (Index_t i = 0; i < length; ++i) {
        Real_t tmpV = vnew[i];
        if (FABS(tmpV - 1.0) < v_cut)
          tmpV = 1.0;
        v[i] = tmpV;
      }
    }
  }

  static inline void LagrangeElements(
      const Real_t *x, const Real_t *y, const Real_t *z, const Real_t *xd,
      const Real_t *yd, const Real_t *zd, Real_t *e, Real_t *p, Real_t *q,
      Real_t *qq, Real_t *ql, Real_t *ss, Real_t *v, Real_t *volo, Real_t *delv,
      Real_t *arealg, Real_t *vdov, const Real_t *elemMass, Index_t **nodelist,
      const Int_t *elemBC, const Index_t *lxim, const Index_t *lxip,
      const Index_t *letam, const Index_t *letap, const Index_t *lzetam,
      const Index_t *lzetap, const Index_t *regElemSize, Index_t numElem,
      Index_t numReg, Index_t sizeX, Index_t sizeY, Index_t sizeZ, Int_t cost,
      Real_t deltatime, Real_t qstop, Real_t monoq_limiter_mult,
      Real_t monoq_max_slope, Real_t qlc_monoq, Real_t qqc_monoq,
      Real_t eosvmin, Real_t eosvmax, Real_t e_cut, Real_t p_cut, Real_t ss4o3,
      Real_t q_cut, Real_t v_cut, Real_t pmin, Real_t emin, Real_t rho0) {
    Real_t *vnew = AllocateReal(numElem);

    CalcLagrangeElements(x, y, z, xd, yd, zd, volo, v, delv, arealg, vdov,
                         nodelist, vnew, deltatime, numElem);
    CalcQForElems(x, y, z, xd, yd, zd, volo, vdov, elemMass, nodelist, elemBC,
                  lxim, lxip, letam, letap, lzetam, lzetap, regElemSize, q, qq,
                  ql, vnew, numElem, numReg, sizeX, sizeY, sizeZ, qstop,
                  monoq_limiter_mult, monoq_max_slope, qlc_monoq, qqc_monoq);
    ApplyMaterialPropertiesForElems(
        e, p, q, qq, ql, ss, v, delv, vnew, regElemSize, numElem, numReg, cost,
        eosvmin, eosvmax, e_cut, p_cut, ss4o3, q_cut, pmin, emin, rho0);
    UpdateVolumesForElems(v, vnew, v_cut, numElem);

    free(vnew);
  }

  //**************************************************
  // Time constraints (simplified)
  //**************************************************

  static inline void CalcCourantConstraintForElems(
      const Real_t *ss, const Real_t *arealg, const Real_t *vdov,
      Index_t length, Real_t qqc, Real_t *dtcourant) {
    Real_t dtcourant_tmp = *dtcourant;
    Real_t qqc2 = 64.0 * qqc * qqc;

    for (Index_t i = 0; i < length; ++i) {
      Real_t dtf = ss[i] * ss[i];

      if (vdov[i] < 0.0) {
        dtf = dtf + qqc2 * arealg[i] * arealg[i] * vdov[i] * vdov[i];
      }

      dtf = SQRT(dtf);
      dtf = arealg[i] / dtf;

      if (vdov[i] != 0.0) {
        if (dtf < dtcourant_tmp) {
          dtcourant_tmp = dtf;
        }
      }
    }

    *dtcourant = dtcourant_tmp;
  }

  static inline void CalcHydroConstraintForElems(
      const Real_t *vdov, Index_t length, Real_t dvovmax, Real_t *dthydro) {
    Real_t dthydro_tmp = *dthydro;

    for (Index_t i = 0; i < length; ++i) {
      if (vdov[i] != 0.0) {
        Real_t dtdvov = dvovmax / (FABS(vdov[i]) + 1.e-20);
        if (dthydro_tmp > dtdvov) {
          dthydro_tmp = dtdvov;
        }
      }
    }

    *dthydro = dthydro_tmp;
  }

  static inline void CalcTimeConstraintsForElems(
      const Real_t *ss, const Real_t *arealg, const Real_t *vdov,
      const Index_t *regElemSize, Index_t numReg, Real_t qqc, Real_t dvovmax,
      Real_t *dtcourant, Real_t *dthydro) {
    *dtcourant = 1.0e+20;
    *dthydro = 1.0e+20;

    for (Index_t r = 0; r < numReg; ++r) {
      CalcCourantConstraintForElems(ss, arealg, vdov, regElemSize[r], qqc,
                                    dtcourant);
      CalcHydroConstraintForElems(vdov, regElemSize[r], dvovmax, dthydro);
    }
  }

  //**************************************************
  // Lagrange Leap Frog
  //**************************************************

  static inline void LagrangeLeapFrog(
      Real_t * x, Real_t * y, Real_t * z, Real_t * xd, Real_t * yd, Real_t * zd,
      Real_t * xdd, Real_t * ydd, Real_t * zdd, Real_t * fx, Real_t * fy,
      Real_t * fz, const Real_t *nodalMass, Real_t *e, Real_t *p, Real_t *q,
      Real_t *qq, Real_t *ql, Real_t *ss, Real_t *v, Real_t *volo, Real_t *delv,
      Real_t *arealg, Real_t *vdov, const Real_t *elemMass, Index_t **nodelist,
      const Int_t *elemBC, const Index_t *lxim, const Index_t *lxip,
      const Index_t *letam, const Index_t *letap, const Index_t *lzetam,
      const Index_t *lzetap, const Index_t *nodeElemStart,
      const Index_t *nodeElemCornerList, const Index_t *regElemSize,
      const Index_t *symmX, const Index_t *symmY, const Index_t *symmZ,
      Index_t symmX_size, Index_t symmY_size, Index_t symmZ_size,
      Index_t numElem, Index_t numNode, Index_t numReg, Index_t sizeX,
      Index_t sizeY, Index_t sizeZ, Int_t cost, Real_t deltatime, Real_t hgcoef,
      Real_t u_cut, Real_t qstop, Real_t monoq_limiter_mult,
      Real_t monoq_max_slope, Real_t qlc_monoq, Real_t qqc_monoq,
      Real_t eosvmin, Real_t eosvmax, Real_t e_cut, Real_t p_cut, Real_t ss4o3,
      Real_t q_cut, Real_t v_cut, Real_t pmin, Real_t emin, Real_t rho0,
      Real_t qqc, Real_t dvovmax, Real_t *dtcourant, Real_t *dthydro) {
    LagrangeNodal(x, y, z, xd, yd, zd, xdd, ydd, zdd, fx, fy, fz, nodalMass,
                  nodelist, volo, v, p, q, ss, elemMass, nodeElemStart,
                  nodeElemCornerList, symmX, symmY, symmZ, symmX_size,
                  symmY_size, symmZ_size, hgcoef, deltatime, u_cut, numElem,
                  numNode, sizeX);
    LagrangeElements(
        x, y, z, xd, yd, zd, e, p, q, qq, ql, ss, v, volo, delv, arealg, vdov,
        elemMass, nodelist, elemBC, lxim, lxip, letam, letap, lzetam, lzetap,
        regElemSize, numElem, numReg, sizeX, sizeY, sizeZ, cost, deltatime,
        qstop, monoq_limiter_mult, monoq_max_slope, qlc_monoq, qqc_monoq,
        eosvmin, eosvmax, e_cut, p_cut, ss4o3, q_cut, v_cut, pmin, emin, rho0);
    CalcTimeConstraintsForElems(ss, arealg, vdov, regElemSize, numReg, qqc,
                                dvovmax, dtcourant, dthydro);
  }

  //**************************************************
  // Command line parsing
  //**************************************************

  static int StrToInt(const char *token, int *retVal) {
    const char *c;
    char *endptr;
    const int decimal_base = 10;

    if (token == NULL)
      return 0;

    c = token;
    *retVal = (int)strtol(c, &endptr, decimal_base);
    if ((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
      return 1;
    else
      return 0;
  }

  static void PrintCommandLineOptions(char *execname, int myRank) {
    if (myRank == 0) {
      printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions (def: 1)\n");
      printf(
          " -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -p              : Print out progress\n");
      printf(" -h              : This message\n\n");
    }
  }

  static void ParseError(const char *message, int myRank) {
    if (myRank == 0) {
      printf("%s\n", message);
      exit(-1);
    }
  }

  static inline void ParseCommandLineOptions(int argc, char *argv[], int myRank,
                                             cmdLineOpts *opts) {
    if (argc > 1) {
      int i = 1;
      while (i < argc) {
        int ok;
        if (strcmp(argv[i], "-i") == 0) {
          if (i + 1 >= argc) {
            ParseError("Missing integer argument to -i", myRank);
          }
          ok = StrToInt(argv[i + 1], &(opts->its));
          if (!ok) {
            ParseError("Parse Error on option -i", myRank);
          }
          i += 2;
        } else if (strcmp(argv[i], "-s") == 0) {
          if (i + 1 >= argc) {
            ParseError("Missing integer argument to -s\n", myRank);
          }
          ok = StrToInt(argv[i + 1], &(opts->nx));
          if (!ok) {
            ParseError("Parse Error on option -s", myRank);
          }
          i += 2;
        } else if (strcmp(argv[i], "-r") == 0) {
          if (i + 1 >= argc) {
            ParseError("Missing integer argument to -r\n", myRank);
          }
          ok = StrToInt(argv[i + 1], &(opts->numReg));
          if (!ok) {
            ParseError("Parse Error on option -r", myRank);
          }
          i += 2;
        } else if (strcmp(argv[i], "-p") == 0) {
          opts->showProg = 1;
          i++;
        } else if (strcmp(argv[i], "-q") == 0) {
          opts->quiet = 1;
          i++;
        } else if (strcmp(argv[i], "-b") == 0) {
          if (i + 1 >= argc) {
            ParseError("Missing integer argument to -b\n", myRank);
          }
          ok = StrToInt(argv[i + 1], &(opts->balance));
          if (!ok) {
            ParseError("Parse Error on option -b", myRank);
          }
          i += 2;
        } else if (strcmp(argv[i], "-c") == 0) {
          if (i + 1 >= argc) {
            ParseError("Missing integer argument to -c\n", myRank);
          }
          ok = StrToInt(argv[i + 1], &(opts->cost));
          if (!ok) {
            ParseError("Parse Error on option -c", myRank);
          }
          i += 2;
        } else if (strcmp(argv[i], "-h") == 0) {
          PrintCommandLineOptions(argv[0], myRank);
          exit(0);
        } else {
          char msg[80];
          PrintCommandLineOptions(argv[0], myRank);
          sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
          ParseError(msg, myRank);
        }
      }
    }
  }

  //**************************************************
  // Main function - Domain struct eliminated
  //**************************************************

  int main(int argc, char *argv[]) {
    Int_t numRanks = 1;
    Int_t myRank = 0;
    cmdLineOpts opts;

    opts.its = 9999999;
    opts.nx = 30;
    opts.numReg = 11;
    opts.showProg = 0;
    opts.quiet = 0;
    opts.balance = 1;
    opts.cost = 1;

    ParseCommandLineOptions(argc, argv, myRank, &opts);

    CARTS_BENCHMARKS_START();

    if ((myRank == 0) && (opts.quiet == 0)) {
      printf("Running problem size %d^3 per domain until completion\n",
             opts.nx);
      printf("Num processors: %d\n", numRanks);
#if _OPENMP
      printf("Num threads: %d\n", omp_get_max_threads());
#endif
      printf("Total number of elements: %lld\n\n",
             (long long int)(numRanks * opts.nx * opts.nx * opts.nx));
      printf("To run other sizes, use -s <integer>.\n");
      printf("To run a fixed number of iterations, use -i <integer>.\n");
      printf("To run a more or less balanced region set, use -b <integer>.\n");
      printf("To change the relative costs of regions, use -c <integer>.\n");
      printf("To print out progress, use -p\n");
      printf("To write an output file for VisIt, use -v\n");
      printf("See help (-h) for more options\n\n");
    }

    CARTS_E2E_TIMER_START("lulesh");

    // Grid dimensions
    Index_t edgeElems = opts.nx;
    Index_t edgeNodes = edgeElems + 1;
    Index_t sizeX = edgeElems;
    Index_t sizeY = edgeElems;
    Index_t sizeZ = edgeElems;
    Index_t numElem = edgeElems * edgeElems * edgeElems;
    Index_t numNode = edgeNodes * edgeNodes * edgeNodes;
    Int_t cost = opts.cost;
    Index_t numReg = opts.numReg;
    Int_t tp = 1; // Single processor
    Index_t colLoc = 0, rowLoc = 0, planeLoc = 0;

    // Node-centered arrays
    Real_t *x = AllocateReal(numNode);
    Real_t *y = AllocateReal(numNode);
    Real_t *z = AllocateReal(numNode);
    Real_t *xd = AllocateReal(numNode);
    Real_t *yd = AllocateReal(numNode);
    Real_t *zd = AllocateReal(numNode);
    Real_t *xdd = AllocateReal(numNode);
    Real_t *ydd = AllocateReal(numNode);
    Real_t *zdd = AllocateReal(numNode);
    Real_t *fx = AllocateReal(numNode);
    Real_t *fy = AllocateReal(numNode);
    Real_t *fz = AllocateReal(numNode);
    Real_t *nodalMass = AllocateReal(numNode);

    // Element-centered arrays
    Index_t **nodelist = AllocateIndex2D(numElem, 8);
    Index_t *lxim = AllocateIndex(numElem);
    Index_t *lxip = AllocateIndex(numElem);
    Index_t *letam = AllocateIndex(numElem);
    Index_t *letap = AllocateIndex(numElem);
    Index_t *lzetam = AllocateIndex(numElem);
    Index_t *lzetap = AllocateIndex(numElem);
    Int_t *elemBC = AllocateInt(numElem);
    Real_t *e = AllocateReal(numElem);
    Real_t *p = AllocateReal(numElem);
    Real_t *q = AllocateReal(numElem);
    Real_t *ql = AllocateReal(numElem);
    Real_t *qq = AllocateReal(numElem);
    Real_t *v = AllocateReal(numElem);
    Real_t *volo = AllocateReal(numElem);
    Real_t *delv = AllocateReal(numElem);
    Real_t *vdov = AllocateReal(numElem);
    Real_t *arealg = AllocateReal(numElem);
    Real_t *ss = AllocateReal(numElem);
    Real_t *elemMass = AllocateReal(numElem);

    // Region arrays
    Index_t *regNumList = AllocateIndex(numElem);
    Index_t *regElemSize = AllocateIndex(numReg);

    // Symmetry arrays (single processor = symmetry on all faces at origin)
    Index_t symmX_size = edgeNodes * edgeNodes;
    Index_t symmY_size = edgeNodes * edgeNodes;
    Index_t symmZ_size = edgeNodes * edgeNodes;
    Index_t *symmX = AllocateIndex(symmX_size);
    Index_t *symmY = AllocateIndex(symmY_size);
    Index_t *symmZ = AllocateIndex(symmZ_size);

    // Thread support structures
    Index_t *nodeElemStart = NULL;
    Index_t *nodeElemCornerList = NULL;

    // Simulation parameters
    Real_t deltatime, time_val = 0.0, stoptime = 1.0e-2;
    Real_t dtfixed = -1.0e-6, dtcourant = 1.0e+20, dthydro = 1.0e+20;
    Real_t dtmax = 1.0e-2;
    Real_t deltatimemultlb = 1.1, deltatimemultub = 1.2;
    Int_t cycle = 0;

    // Initialize node positions
    Index_t meshEdgeElems = tp * opts.nx;
    Index_t nidx = 0;
    for (Index_t pl = 0; pl < edgeNodes; ++pl) {
      Real_t tz = 1.125 * (planeLoc * opts.nx + pl) / meshEdgeElems;
      for (Index_t rw = 0; rw < edgeNodes; ++rw) {
        Real_t ty = 1.125 * (rowLoc * opts.nx + rw) / meshEdgeElems;
        for (Index_t cl = 0; cl < edgeNodes; ++cl) {
          Real_t tx = 1.125 * (colLoc * opts.nx + cl) / meshEdgeElems;
          x[nidx] = tx;
          y[nidx] = ty;
          z[nidx] = tz;
          ++nidx;
        }
      }
    }

    // Build element connectivity
    Index_t zidx = 0;
    nidx = 0;
    for (Index_t pl = 0; pl < edgeElems; ++pl) {
      for (Index_t rw = 0; rw < edgeElems; ++rw) {
        for (Index_t cl = 0; cl < edgeElems; ++cl) {
          nodelist[zidx][0] = nidx;
          nodelist[zidx][1] = nidx + 1;
          nodelist[zidx][2] = nidx + edgeNodes + 1;
          nodelist[zidx][3] = nidx + edgeNodes;
          nodelist[zidx][4] = nidx + edgeNodes * edgeNodes;
          nodelist[zidx][5] = nidx + edgeNodes * edgeNodes + 1;
          nodelist[zidx][6] = nidx + edgeNodes * edgeNodes + edgeNodes + 1;
          nodelist[zidx][7] = nidx + edgeNodes * edgeNodes + edgeNodes;
          ++zidx;
          ++nidx;
        }
        ++nidx;
      }
      nidx += edgeNodes;
    }

    // Initialize element arrays
    for (Index_t i = 0; i < numElem; ++i) {
      e[i] = 0.0;
      p[i] = 0.0;
      q[i] = 0.0;
      ss[i] = 0.0;
      v[i] = 1.0;
    }

    // Initialize node velocities/accelerations/masses
    for (Index_t i = 0; i < numNode; ++i) {
      xd[i] = 0.0;
      yd[i] = 0.0;
      zd[i] = 0.0;
      xdd[i] = 0.0;
      ydd[i] = 0.0;
      zdd[i] = 0.0;
      nodalMass[i] = 0.0;
    }

    // Setup thread support structures (always needed)
    Index_t *nodeElemCount = AllocateIndex(numNode);
    for (Index_t i = 0; i < numNode; ++i)
      nodeElemCount[i] = 0;
    for (Index_t i = 0; i < numElem; ++i) {
      for (Index_t j = 0; j < 8; ++j)
        ++(nodeElemCount[nodelist[i][j]]);
    }
    nodeElemStart = AllocateIndex(numNode + 1);
    nodeElemStart[0] = 0;
    for (Index_t i = 1; i <= numNode; ++i)
      nodeElemStart[i] = nodeElemStart[i - 1] + nodeElemCount[i - 1];
    nodeElemCornerList = AllocateIndex(nodeElemStart[numNode]);
    for (Index_t i = 0; i < numNode; ++i)
      nodeElemCount[i] = 0;
    for (Index_t i = 0; i < numElem; ++i) {
      for (Index_t j = 0; j < 8; ++j) {
        Index_t m = nodelist[i][j];
        Index_t k = i * 8 + j;
        Index_t offset = nodeElemStart[m] + nodeElemCount[m];
        nodeElemCornerList[offset] = k;
        ++(nodeElemCount[m]);
      }
    }
    free(nodeElemCount);

    // Setup regions (single region for simplicity)
    for (Index_t i = 0; i < numElem; i++)
      regNumList[i] = 1;
    regElemSize[0] = numElem;
    for (Index_t r = 1; r < numReg; ++r) {
      regElemSize[r] = 0;
    }

    // Setup symmetry planes
    nidx = 0;
    for (Index_t i = 0; i < edgeNodes; ++i) {
      Index_t planeInc = i * edgeNodes * edgeNodes;
      Index_t rowInc = i * edgeNodes;
      for (Index_t j = 0; j < edgeNodes; ++j) {
        symmZ[nidx] = rowInc + j;
        symmY[nidx] = planeInc + j;
        symmX[nidx] = planeInc + j * edgeNodes;
        ++nidx;
      }
    }

    // Setup element connectivities
    lxim[0] = 0;
    for (Index_t i = 1; i < numElem; ++i) {
      lxim[i] = i - 1;
      lxip[i - 1] = i;
    }
    lxip[numElem - 1] = numElem - 1;

    for (Index_t i = 0; i < edgeElems; ++i) {
      letam[i] = i;
      letap[numElem - edgeElems + i] = numElem - edgeElems + i;
    }
    for (Index_t i = edgeElems; i < numElem; ++i) {
      letam[i] = i - edgeElems;
      letap[i - edgeElems] = i;
    }

    for (Index_t i = 0; i < edgeElems * edgeElems; ++i) {
      lzetam[i] = i;
      lzetap[numElem - edgeElems * edgeElems + i] =
          numElem - edgeElems * edgeElems + i;
    }
    for (Index_t i = edgeElems * edgeElems; i < numElem; ++i) {
      lzetam[i] = i - edgeElems * edgeElems;
      lzetap[i - edgeElems * edgeElems] = i;
    }

    // Setup boundary conditions (origin is symmetric, others free)
    for (Index_t i = 0; i < numElem; ++i)
      elemBC[i] = 0;
    for (Index_t i = 0; i < edgeElems; ++i) {
      Index_t planeInc = i * edgeElems * edgeElems;
      Index_t rowInc = i * edgeElems;
      for (Index_t j = 0; j < edgeElems; ++j) {
        elemBC[rowInc + j] |= ZETA_M_SYMM;
        elemBC[rowInc + j + numElem - edgeElems * edgeElems] |= ZETA_P_FREE;
        elemBC[planeInc + j] |= ETA_M_SYMM;
        elemBC[planeInc + (edgeElems - 1) * edgeElems + j] |= ETA_P_FREE;
        elemBC[planeInc + j * edgeElems] |= XI_M_SYMM;
        elemBC[planeInc + j * edgeElems + edgeElems - 1] |= XI_P_FREE;
      }
    }

    // Initialize field data (volumes, masses)
    for (Index_t i = 0; i < numElem; ++i) {
      Real_t x_local[8], y_local[8], z_local[8];
      for (Index_t lnode = 0; lnode < 8; ++lnode) {
        Index_t gnode = nodelist[i][lnode];
        x_local[lnode] = x[gnode];
        y_local[lnode] = y[gnode];
        z_local[lnode] = z[gnode];
      }
      Real_t volume = CalcElemVolume(x_local, y_local, z_local);
      volo[i] = volume;
      elemMass[i] = volume;
      for (Index_t j = 0; j < 8; ++j) {
        Index_t idx = nodelist[i][j];
        nodalMass[idx] += volume / 8.0;
      }
    }

    // Deposit initial energy
    const Real_t ebase = 3.948746e+7;
    Real_t scale = (opts.nx * tp) / 45.0;
    Real_t einit = ebase * scale * scale * scale;
    e[0] = einit;
    deltatime = (0.5 * cbrt(volo[0])) / sqrt(2.0 * einit);

    // Main timestep loop
    struct timeval start, end;
    gettimeofday(&start, NULL);

    CARTS_KERNEL_TIMER_START("lulesh");
    while ((time_val < stoptime) && (cycle < opts.its)) {
      TimeIncrement(&deltatime, &time_val, &cycle, stoptime, dtfixed, dtcourant,
                    dthydro, deltatimemultlb, deltatimemultub, dtmax);

      printf("iteration %d, delta time %f, energy %f\n", cycle, deltatime,
             e[0]);
      fflush(stdout);

      LagrangeLeapFrog(
          x, y, z, xd, yd, zd, xdd, ydd, zdd, fx, fy, fz, nodalMass, e, p, q,
          qq, ql, ss, v, volo, delv, arealg, vdov, elemMass, nodelist, elemBC,
          lxim, lxip, letam, letap, lzetam, lzetap, nodeElemStart,
          nodeElemCornerList, regElemSize, symmX, symmY, symmZ, symmX_size,
          symmY_size, symmZ_size, numElem, numNode, numReg, sizeX, sizeY, sizeZ,
          cost, deltatime, c_hgcoef, c_u_cut, c_qstop, c_monoq_limiter_mult,
          c_monoq_max_slope, c_qlc_monoq, c_qqc_monoq, c_eosvmin, c_eosvmax,
          c_e_cut, c_p_cut, c_ss4o3, c_q_cut, c_v_cut, c_pmin, c_emin,
          c_refdens, c_qqc, c_dvovmax, &dtcourant, &dthydro);

      if ((opts.showProg != 0) && (opts.quiet == 0)) {
        printf("cycle = %d, time = %e, dt=%e\n", cycle, time_val, deltatime);
      }
      CARTS_KERNEL_TIMER_ACCUM("lulesh");
    }

    gettimeofday(&end, NULL);
    double elapsed_time = (double)(end.tv_sec - start.tv_sec) +
                          ((double)(end.tv_usec - start.tv_usec)) / 1000000;

    CARTS_KERNEL_TIMER_PRINT("lulesh");
    CARTS_E2E_TIMER_STOP();
    CARTS_BENCHMARKS_STOP();

    // Output results
    Real_t grindTime1 =
        ((elapsed_time * 1e6) / cycle) / (opts.nx * opts.nx * opts.nx);
    Real_t grindTime2 = ((elapsed_time * 1e6) / cycle) /
                        (opts.nx * opts.nx * opts.nx * numRanks);

    printf("Run completed:  \n");
    printf("   Problem size        =  %i \n", opts.nx);
    printf("   MPI tasks           =  %i \n", numRanks);
    printf("   Iteration count     =  %i \n", cycle);
    printf("   Final Origin Energy = %12.6e \n", e[0]);
    printf("checksum: %.6e\n", e[0]);

    Real_t MaxAbsDiff = 0.0;
    Real_t TotalAbsDiff = 0.0;
    Real_t MaxRelDiff = 0.0;

    for (Index_t j = 0; j < opts.nx; ++j) {
      for (Index_t k = j + 1; k < opts.nx; ++k) {
        Real_t AbsDiff = FABS(e[j * opts.nx + k] - e[k * opts.nx + j]);
        TotalAbsDiff += AbsDiff;
        if (MaxAbsDiff < AbsDiff)
          MaxAbsDiff = AbsDiff;
        Real_t RelDiff = AbsDiff / e[k * opts.nx + j];
        if (MaxRelDiff < RelDiff)
          MaxRelDiff = RelDiff;
      }
    }

    printf("   Testing Plane 0 of Energy Array on rank 0:\n");
    printf("        MaxAbsDiff   = %12.6e\n", MaxAbsDiff);
    printf("        TotalAbsDiff = %12.6e\n", TotalAbsDiff);
    printf("        MaxRelDiff   = %12.6e\n\n", MaxRelDiff);

    printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
    printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n",
           grindTime1, grindTime2);
    printf("FOM                  = %10.8g (z/s)\n\n", 1000.0 / grindTime2);

    // Free arrays
    free(x);
    free(y);
    free(z);
    free(xd);
    free(yd);
    free(zd);
    free(xdd);
    free(ydd);
    free(zdd);
    free(fx);
    free(fy);
    free(fz);
    free(nodalMass);
    free(lxim);
    free(lxip);
    free(letam);
    free(letap);
    free(lzetam);
    free(lzetap);
    free(elemBC);
    free(e);
    free(p);
    free(q);
    free(ql);
    free(qq);
    free(v);
    free(volo);
    free(delv);
    free(vdov);
    free(arealg);
    free(ss);
    free(elemMass);
    free(regNumList);
    free(regElemSize);
    free(symmX);
    free(symmY);
    free(symmZ);
    free(nodeElemStart);
    free(nodeElemCornerList);

    FreeIndex2D(nodelist, numElem);

    return 0;
  }

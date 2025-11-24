/**********************************************************************************************/
/*  Standalone CARTS test version of SparseLU with tasks
 *  Self-contained single-file version for CARTS compiler testing
 *
 *  Based on Barcelona OpenMP Tasks Suite
 *  Copyright (C) 2009 Barcelona Supercomputing Center
 *  License: GNU GPL
 */
/**********************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 1.0E-6

//===----------------------------------------------------------------------===//
// Utility Functions (Static)
//===----------------------------------------------------------------------===//

static float **allocate_clean_block(int submatrix_size) {
  float **block = (float **)malloc(submatrix_size * sizeof(float *));
  if (block == NULL) {
    fprintf(stderr, "Error: malloc failed in allocate_clean_block\n");
    exit(101);
  }
  for (int i = 0; i < submatrix_size; i++) {
    block[i] = (float *)malloc(submatrix_size * sizeof(float));
    if (block[i] == NULL) {
      fprintf(stderr, "Error: malloc failed in allocate_clean_block\n");
      exit(101);
    }
    for (int j = 0; j < submatrix_size; j++) {
      block[i][j] = 0.0;
    }
  }
  return block;
}

static void free_block(float **block, int submatrix_size) {
  if (block) {
    for (int i = 0; i < submatrix_size; i++) {
      free(block[i]);
    }
    free(block);
  }
}

static void lu0(float **diag, int submatrix_size) {
  int i, j, k;

  for (k = 0; k < submatrix_size; k++)
    for (i = k + 1; i < submatrix_size; i++) {
      diag[i][k] = diag[i][k] / diag[k][k];
      for (j = k + 1; j < submatrix_size; j++)
        diag[i][j] = diag[i][j] - diag[i][k] * diag[k][j];
    }
}

static void bdiv(float **diag, float **row, int submatrix_size) {
  int i, j, k;
  for (i = 0; i < submatrix_size; i++)
    for (k = 0; k < submatrix_size; k++) {
      row[i][k] = row[i][k] / diag[k][k];
      for (j = k + 1; j < submatrix_size; j++)
        row[i][j] = row[i][j] - row[i][k] * diag[k][j];
    }
}

static void bmod(float **row, float **col, float **inner, int submatrix_size) {
  int i, j, k;
  for (i = 0; i < submatrix_size; i++)
    for (j = 0; j < submatrix_size; j++)
      for (k = 0; k < submatrix_size; k++)
        inner[i][j] = inner[i][j] - row[i][k] * col[k][j];
}

static void fwd(float **diag, float **col, int submatrix_size) {
  int i, j, k;
  for (j = 0; j < submatrix_size; j++)
    for (k = 0; k < submatrix_size; k++)
      for (i = k + 1; i < submatrix_size; i++)
        col[i][j] = col[i][j] - diag[i][k] * col[k][j];
}

//===----------------------------------------------------------------------===//
// Matrix Generation
//===----------------------------------------------------------------------===//

static void genmat(float ***M, int matrix_size, int submatrix_size) {
  int null_entry, init_val, i, j, ii, jj;

  init_val = 1325;

  /* generating the structure */
  for (ii = 0; ii < matrix_size; ii++) {
    for (jj = 0; jj < matrix_size; jj++) {
#pragma omp task shared(M)
      {
        /* computing null entries */
        null_entry = 0;
        if ((ii < jj) && (ii % 3 != 0))
          null_entry = 1;
        if ((ii > jj) && (jj % 3 != 0))
          null_entry = 1;
        if (ii % 2 == 1)
          null_entry = 1;
        if (jj % 2 == 1)
          null_entry = 1;
        if (ii == jj)
          null_entry = 0;
        if (ii == jj - 1)
          null_entry = 0;
        if (ii - 1 == jj)
          null_entry = 0;
        /* allocating matrix */
        if (null_entry == 0) {
          M[ii][jj] = allocate_clean_block(submatrix_size);
          /* initializing matrix */
          for (i = 0; i < submatrix_size; i++) {
            for (j = 0; j < submatrix_size; j++) {
              init_val = (3125 * init_val) % 65536;
              M[ii][jj][i][j] = (float)((init_val - 32768.0) / 16384.0);
            }
          }
        } else {
          M[ii][jj] = NULL;
        }
      }
    }
  }
#pragma omp taskwait
}

static void sparselu_init(float ****pBENCH, int matrix_size,
                          int submatrix_size) {
  *pBENCH = (float ***)malloc(matrix_size * sizeof(float **));
  if (*pBENCH == NULL) {
    fprintf(stderr, "Error: malloc failed for benchmark matrix\n");
    exit(101);
  }
  for (int i = 0; i < matrix_size; i++) {
    (*pBENCH)[i] = (float **)malloc(matrix_size * sizeof(float *));
    if ((*pBENCH)[i] == NULL) {
      fprintf(stderr, "Error: malloc failed for benchmark matrix\n");
      exit(101);
    }
  }
  genmat(*pBENCH, matrix_size, submatrix_size);
}

//===----------------------------------------------------------------------===//
// Parallel SparseLU with Tasks
//===----------------------------------------------------------------------===//

static void sparselu_par_call(float ***BENCH, int matrix_size,
                              int submatrix_size) {
  int ii, jj, kk;

#pragma omp parallel
#pragma omp single nowait
  for (kk = 0; kk < matrix_size; kk++) {
    lu0(BENCH[kk][kk], submatrix_size);
    for (jj = kk + 1; jj < matrix_size; jj++)
      if (BENCH[kk][jj] != NULL)
#pragma omp task untied firstprivate(kk, jj) shared(BENCH)
      {
        fwd(BENCH[kk][kk], BENCH[kk][jj], submatrix_size);
      }
    for (ii = kk + 1; ii < matrix_size; ii++)
      if (BENCH[ii][kk] != NULL)
#pragma omp task untied firstprivate(kk, ii) shared(BENCH)
      {
        bdiv(BENCH[kk][kk], BENCH[ii][kk], submatrix_size);
      }

#pragma omp taskwait

    for (ii = kk + 1; ii < matrix_size; ii++)
      if (BENCH[ii][kk] != NULL)
        for (jj = kk + 1; jj < matrix_size; jj++)
          if (BENCH[kk][jj] != NULL)
#pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
          {
            if (BENCH[ii][jj] == NULL)
              BENCH[ii][jj] = allocate_clean_block(submatrix_size);
            bmod(BENCH[ii][kk], BENCH[kk][jj], BENCH[ii][jj], submatrix_size);
          }

#pragma omp taskwait
  }
}

static void sparselu_seq_call(float ***BENCH, int matrix_size,
                              int submatrix_size) {
  // Simple sequential implementation for verification
  int ii, jj, kk;
  for (kk = 0; kk < matrix_size; kk++) {
    if (BENCH[kk][kk] != NULL) {
      lu0(BENCH[kk][kk], submatrix_size);
    }
    for (jj = kk + 1; jj < matrix_size; jj++) {
      if (BENCH[kk][jj] != NULL && BENCH[kk][kk] != NULL) {
        fwd(BENCH[kk][kk], BENCH[kk][jj], submatrix_size);
      }
    }
    for (ii = kk + 1; ii < matrix_size; ii++) {
      if (BENCH[ii][kk] != NULL && BENCH[kk][kk] != NULL) {
        bdiv(BENCH[kk][kk], BENCH[ii][kk], submatrix_size);
      }
    }
    for (ii = kk + 1; ii < matrix_size; ii++) {
      if (BENCH[ii][kk] != NULL) {
        for (jj = kk + 1; jj < matrix_size; jj++) {
          if (BENCH[kk][jj] != NULL) {
            if (BENCH[ii][jj] == NULL) {
              BENCH[ii][jj] = allocate_clean_block(submatrix_size);
            }
            bmod(BENCH[ii][kk], BENCH[kk][jj], BENCH[ii][jj], submatrix_size);
          }
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Main Test Function
//===----------------------------------------------------------------------===//

int main(void) {
  float ***BENCH, ***BENCH_SEQ;
#ifdef SIZE
  int matrix_size = SIZE / 64; // Reasonable block count based on SIZE
  if (matrix_size < 4)
    matrix_size = 4; // Minimum size
  if (matrix_size > 64)
    matrix_size = 64; // Maximum size
#else
  int matrix_size = 16; // Default for testing
#endif
  int submatrix_size = 8; // Fixed subblock size

  printf("SparseLU Task Test (CARTS)\n");
  printf("Matrix size: %d x %d blocks\n", matrix_size, matrix_size);
  printf("Submatrix size: %d x %d\n", submatrix_size, submatrix_size);

  // Initialize matrix for parallel version
#pragma omp parallel
#pragma omp master
  sparselu_init(&BENCH, matrix_size, submatrix_size);

  // Save initial state for verification
  BENCH_SEQ = (float ***)malloc(matrix_size * sizeof(float **));
  for (int i = 0; i < matrix_size; i++) {
    BENCH_SEQ[i] = (float **)malloc(matrix_size * sizeof(float *));
    for (int j = 0; j < matrix_size; j++) {
      if (BENCH[i][j] != NULL) {
        BENCH_SEQ[i][j] = allocate_clean_block(submatrix_size);
        for (int row = 0; row < submatrix_size; row++) {
          for (int col = 0; col < submatrix_size; col++) {
            BENCH_SEQ[i][j][row][col] = BENCH[i][j][row][col];
          }
        }
      } else {
        BENCH_SEQ[i][j] = NULL;
      }
    }
  }

  printf("Running parallel SparseLU with tasks...\n");

  // Run parallel SparseLU
  sparselu_par_call(BENCH, matrix_size, submatrix_size);

  printf("Running sequential SparseLU for verification...\n");

  // Run sequential version (simple implementation for now)
  sparselu_seq_call(BENCH_SEQ, matrix_size, submatrix_size);

  // Compare results
  printf("Verifying results...\n");
  double error = 0.0;
  int count = 0;
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      if (BENCH[i][j] != NULL && BENCH_SEQ[i][j] != NULL) {
        for (int row = 0; row < submatrix_size; row++) {
          for (int col = 0; col < submatrix_size; col++) {
            double diff = BENCH[i][j][row][col] - BENCH_SEQ[i][j][row][col];
            error += diff * diff;
            count++;
          }
        }
      }
    }
  }
  error = sqrt(error / count);

  printf("Verification: %s (RMS error: %.2e)\n",
         (error < 1e-4) ? "PASS" : "FAIL", error);

  // Cleanup
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      if (BENCH[i][j] != NULL)
        free_block(BENCH[i][j], submatrix_size);
      if (BENCH_SEQ[i][j] != NULL)
        free_block(BENCH_SEQ[i][j], submatrix_size);
    }
    free(BENCH[i]);
    free(BENCH_SEQ[i]);
  }
  free(BENCH);
  free(BENCH_SEQ);

  return (error < 1e-4) ? 0 : 1;
}

/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de
 * Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya */
/*                                                                                            */
/**********************************************************************************************/

/*
 * Copyright (c) 1996 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use, copy, modify, and distribute the Software without
 * restriction, provided the Software, including any modified copies made
 * under this license, is not distributed for a fee, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE MASSACHUSETTS INSTITUTE OF TECHNOLOGY BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * /WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Massachusetts
 * Institute of Technology shall not be used in advertising or otherwise
 * to promote the sale, use or other dealings in this Software without
 * prior written authorization from the Massachusetts Institute of
 * Technology.
 *
 */

#include "strassen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to allocate a 2D matrix
static double **alloc_matrix_2d(unsigned size) {
  double **M = (double **)malloc(size * sizeof(double *));
  if (!M) return NULL;
  for (unsigned i = 0; i < size; i++) {
    M[i] = (double *)malloc(size * sizeof(double));
    if (!M[i]) {
      for (unsigned j = 0; j < i; j++) free(M[j]);
      free(M);
      return NULL;
    }
  }
  return M;
}

// Helper function to free a 2D matrix
static void free_matrix_2d(double **M, unsigned size) {
  if (M) {
    for (unsigned i = 0; i < size; i++) {
      free(M[i]);
    }
    free(M);
  }
}

// Helper function to get a submatrix view (returns array of pointers into original matrix)
static double **get_submatrix_view(double **M, unsigned row_start, unsigned col_start, unsigned size) {
  double **view = (double **)malloc(size * sizeof(double *));
  if (!view) return NULL;
  for (unsigned i = 0; i < size; i++) {
    view[i] = M[row_start + i] + col_start;
  }
  return view;
}

// Helper function to free a submatrix view (just frees the pointer array, not the data)
static void free_submatrix_view(double **view) {
  free(view);
}

/*****************************************************************************
 **
 ** OptimizedStrassenMultiply
 **
 ** For large matrices A, B, and C of size MatrixSize * MatrixSize this
 ** function performs the operation C = A x B efficiently.
 **
 ** INPUT:
 **    C = Output matrix (modified in-place)
 **    A = Input matrix A (read-only)
 **    B = Input matrix B (read-only)
 **    MatrixSize = Size of matrices (for n*n matrix, MatrixSize = n)
 **    Depth = Current recursion depth
 **    cutoff_depth = Maximum recursion depth
 **    cutoff_size = Size below which to use base case
 **
 ** OUTPUT:
 **    C = Matrix C contains A x B
 **
 *****************************************************************************/
static void OptimizedStrassenMultiply_par(
    double **C, double **A, double **B, unsigned MatrixSize,
    unsigned int Depth, unsigned int cutoff_depth, unsigned cutoff_size) {
  unsigned QuadrantSize = MatrixSize >> 1; /* MatrixSize / 2 */
  unsigned Column, Row;

  if (MatrixSize <= cutoff_size) {
    // Base case: use simple matrix multiplication
    for (unsigned i = 0; i < MatrixSize; i++) {
      for (unsigned j = 0; j < MatrixSize; j++) {
        C[i][j] = 0.0;
        for (unsigned k = 0; k < MatrixSize; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return;
  }

  // Allocate temporary matrices as 2D arrays
  double **S1 = alloc_matrix_2d(QuadrantSize);
  double **S2 = alloc_matrix_2d(QuadrantSize);
  double **S3 = alloc_matrix_2d(QuadrantSize);
  double **S4 = alloc_matrix_2d(QuadrantSize);
  double **S5 = alloc_matrix_2d(QuadrantSize);
  double **S6 = alloc_matrix_2d(QuadrantSize);
  double **S7 = alloc_matrix_2d(QuadrantSize);
  double **S8 = alloc_matrix_2d(QuadrantSize);
  double **M2 = alloc_matrix_2d(QuadrantSize);
  double **M5 = alloc_matrix_2d(QuadrantSize);
  double **T1sMULT = alloc_matrix_2d(QuadrantSize);
  double **T2sMULT = alloc_matrix_2d(QuadrantSize);

  if (!S1 || !S2 || !S3 || !S4 || !S5 || !S6 || !S7 || !S8 || !M2 || !M5 || !T1sMULT || !T2sMULT) {
    fprintf(stderr, "Memory allocation failed in Strassen\n");
    return;
  }

  if (Depth < cutoff_depth) {

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S1[Row][Column] =
            A[QuadrantSize + Row][Column] + A[QuadrantSize + Row][QuadrantSize + Column];

#pragma omp taskwait

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S2[Row][Column] =
            S1[Row][Column] - A[Row][Column];

#pragma omp taskwait

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S4[Row][Column] =
            A[Row][QuadrantSize + Column] - S2[Row][Column];

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S5[Row][Column] =
            B[Row][QuadrantSize + Column] - B[Row][Column];

#pragma omp taskwait

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S6[Row][Column] =
            B[QuadrantSize + Row][QuadrantSize + Column] - S5[Row][Column];

#pragma omp taskwait

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S8[Row][Column] =
            S6[Row][Column] - B[QuadrantSize + Row][Column];

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S3[Row][Column] =
            A[Row][Column] - A[QuadrantSize + Row][Column];

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S7[Row][Column] =
            B[QuadrantSize + Row][QuadrantSize + Column] - B[Row][QuadrantSize + Column];

#pragma omp taskwait

    // Create views for matrix quadrants
    double **A11_view = get_submatrix_view(A, 0, 0, QuadrantSize);
    double **A12_view = get_submatrix_view(A, 0, QuadrantSize, QuadrantSize);
    double **A21_view = get_submatrix_view(A, QuadrantSize, 0, QuadrantSize);
    double **A22_view = get_submatrix_view(A, QuadrantSize, QuadrantSize, QuadrantSize);
    double **B11_view = get_submatrix_view(B, 0, 0, QuadrantSize);
    double **B12_view = get_submatrix_view(B, 0, QuadrantSize, QuadrantSize);
    double **B21_view = get_submatrix_view(B, QuadrantSize, 0, QuadrantSize);
    double **B22_view = get_submatrix_view(B, QuadrantSize, QuadrantSize, QuadrantSize);
    double **C11_view = get_submatrix_view(C, 0, 0, QuadrantSize);
    double **C12_view = get_submatrix_view(C, 0, QuadrantSize, QuadrantSize);
    double **C21_view = get_submatrix_view(C, QuadrantSize, 0, QuadrantSize);
    double **C22_view = get_submatrix_view(C, QuadrantSize, QuadrantSize, QuadrantSize);

    if (!A11_view || !A12_view || !A21_view || !A22_view ||
        !B11_view || !B12_view || !B21_view || !B22_view ||
        !C11_view || !C12_view || !C21_view || !C22_view) {
      fprintf(stderr, "Memory allocation failed for submatrix views\n");
      return;
    }

    /* M2 = A11 x B11 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(M2, A11_view, B11_view, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    /* M5 = S1 * S5 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    /* Step 1 of T1 = S2 x S6 + M2 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(T1sMULT, S2, S6, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    /* Step 1 of T2 = T1 + S3 x S7 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(T2sMULT, S3, S7, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    /* Step 1 of C11 = M2 + A12 * B21 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(C11_view, A12_view, B21_view, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    /* Step 1 of C12 = S4 x B22 + T1 + M5 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(C12_view, S4, B22_view, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    /* Step 1 of C21 = T2 - A22 * S8 */
#pragma omp task untied
    OptimizedStrassenMultiply_par(C21_view, A22_view, S8, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

#pragma omp taskwait

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column += 1)
        C[Row][Column] += M2[Row][Column];

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column += 1)
        C[Row][QuadrantSize + Column] += M5[Row][Column] +
                                         T1sMULT[Row][Column] +
                                         M2[Row][Column];

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column += 1)
        C[QuadrantSize + Row][Column] = -C[QuadrantSize + Row][Column] +
                                        C[QuadrantSize + Row][QuadrantSize + Column] +
                                        T1sMULT[Row][Column] +
                                        M2[Row][Column];

#pragma omp taskwait

#pragma omp task private(Row, Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column += 1)
        C[QuadrantSize + Row][QuadrantSize + Column] += M5[Row][Column] +
                                         T1sMULT[Row][Column] +
                                         M2[Row][Column];

#pragma omp taskwait

    // Free submatrix views
    free_submatrix_view(A11_view);
    free_submatrix_view(A12_view);
    free_submatrix_view(A21_view);
    free_submatrix_view(A22_view);
    free_submatrix_view(B11_view);
    free_submatrix_view(B12_view);
    free_submatrix_view(B21_view);
    free_submatrix_view(B22_view);
    free_submatrix_view(C11_view);
    free_submatrix_view(C12_view);
    free_submatrix_view(C21_view);
    free_submatrix_view(C22_view);
  } else {
    // Sequential case - same computation but without tasks
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++) {
        S1[Row][Column] =
            A[QuadrantSize + Row][Column] + A[QuadrantSize + Row][QuadrantSize + Column];
        S2[Row][Column] =
            S1[Row][Column] - A[Row][Column];
        S4[Row][Column] =
            A[Row][QuadrantSize + Column] - S2[Row][Column];
        S5[Row][Column] =
            B[Row][QuadrantSize + Column] - B[Row][Column];
        S6[Row][Column] =
            B[QuadrantSize + Row][QuadrantSize + Column] - S5[Row][Column];
        S8[Row][Column] =
            S6[Row][Column] - B[QuadrantSize + Row][Column];
        S3[Row][Column] =
            A[Row][Column] - A[QuadrantSize + Row][Column];
        S7[Row][Column] =
            B[QuadrantSize + Row][QuadrantSize + Column] - B[Row][QuadrantSize + Column];
      }

    // Create views for matrix quadrants
    double **A11_view = get_submatrix_view(A, 0, 0, QuadrantSize);
    double **A12_view = get_submatrix_view(A, 0, QuadrantSize, QuadrantSize);
    double **A21_view = get_submatrix_view(A, QuadrantSize, 0, QuadrantSize);
    double **A22_view = get_submatrix_view(A, QuadrantSize, QuadrantSize, QuadrantSize);
    double **B11_view = get_submatrix_view(B, 0, 0, QuadrantSize);
    double **B12_view = get_submatrix_view(B, 0, QuadrantSize, QuadrantSize);
    double **B21_view = get_submatrix_view(B, QuadrantSize, 0, QuadrantSize);
    double **B22_view = get_submatrix_view(B, QuadrantSize, QuadrantSize, QuadrantSize);
    double **C11_view = get_submatrix_view(C, 0, 0, QuadrantSize);
    double **C12_view = get_submatrix_view(C, 0, QuadrantSize, QuadrantSize);
    double **C21_view = get_submatrix_view(C, QuadrantSize, 0, QuadrantSize);
    double **C22_view = get_submatrix_view(C, QuadrantSize, QuadrantSize, QuadrantSize);

    if (!A11_view || !A12_view || !A21_view || !A22_view ||
        !B11_view || !B12_view || !B21_view || !B22_view ||
        !C11_view || !C12_view || !C21_view || !C22_view) {
      fprintf(stderr, "Memory allocation failed for submatrix views\n");
      return;
    }

    /* M2 = A11 x B11 */
    OptimizedStrassenMultiply_par(M2, A11_view, B11_view, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);
    /* M5 = S1 * S5 */
    OptimizedStrassenMultiply_par(M5, S1, S5, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);
    /* Step 1 of T1 = S2 x S6 + M2 */
    OptimizedStrassenMultiply_par(T1sMULT, S2, S6, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);
    /* Step 1 of T2 = T1 + S3 x S7 */
    OptimizedStrassenMultiply_par(T2sMULT, S3, S7, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);
    /* Step 1 of C11 = M2 + A12 * B21 */
    OptimizedStrassenMultiply_par(C11_view, A12_view, B21_view, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);
    /* Step 1 of C12 = S4 x B22 + T1 + M5 */
    OptimizedStrassenMultiply_par(C12_view, S4, B22_view, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);
    /* Step 1 of C21 = T2 - A22 * S8 */
    OptimizedStrassenMultiply_par(C21_view, A22_view, S8, QuadrantSize,
                                  Depth + 1, cutoff_depth, cutoff_size);

    for (Row = 0; Row < QuadrantSize; Row++) {
      for (Column = 0; Column < QuadrantSize; Column += 1) {
        C[Row][Column] += M2[Row][Column];
        C[Row][QuadrantSize + Column] += M5[Row][Column] +
                                         T1sMULT[Row][Column] +
                                         M2[Row][Column];
        C[QuadrantSize + Row][Column] = -C[QuadrantSize + Row][Column] +
                                        C[QuadrantSize + Row][QuadrantSize + Column] +
                                        T1sMULT[Row][Column] +
                                        M2[Row][Column];
        C[QuadrantSize + Row][QuadrantSize + Column] += M5[Row][Column] +
                                         T1sMULT[Row][Column] +
                                         M2[Row][Column];
      }
    }

    // Free submatrix views
    free_submatrix_view(A11_view);
    free_submatrix_view(A12_view);
    free_submatrix_view(A21_view);
    free_submatrix_view(A22_view);
    free_submatrix_view(B11_view);
    free_submatrix_view(B12_view);
    free_submatrix_view(B21_view);
    free_submatrix_view(B22_view);
    free_submatrix_view(C11_view);
    free_submatrix_view(C12_view);
    free_submatrix_view(C21_view);
    free_submatrix_view(C22_view);
  }

  // Free temporary matrices
  free_matrix_2d(S1, QuadrantSize);
  free_matrix_2d(S2, QuadrantSize);
  free_matrix_2d(S3, QuadrantSize);
  free_matrix_2d(S4, QuadrantSize);
  free_matrix_2d(S5, QuadrantSize);
  free_matrix_2d(S6, QuadrantSize);
  free_matrix_2d(S7, QuadrantSize);
  free_matrix_2d(S8, QuadrantSize);
  free_matrix_2d(M2, QuadrantSize);
  free_matrix_2d(M5, QuadrantSize);
  free_matrix_2d(T1sMULT, QuadrantSize);
  free_matrix_2d(T2sMULT, QuadrantSize);
}

void strassen_main_par(double **A, double **B, double **C, int n,
                       unsigned int cutoff_size, unsigned int cutoff_depth) {
#pragma omp parallel
#pragma omp master
  OptimizedStrassenMultiply_par(C, A, B, n, 1, cutoff_depth, cutoff_size);
}

void strassen_main_seq(double **A, double **B, double **C, int n,
                       unsigned int cutoff_size) {
  // Simple sequential matrix multiplication for verification
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i][j] = 0.0;
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main(void) {
#ifdef SIZE
  int n = SIZE;
  // Ensure size is a power of 2 for Strassen
  int size = 1;
  while (size < n) size *= 2;
  if (size != n) n = size; // Round up to nearest power of 2
  if (n < 16) n = 16;      // Minimum size
  if (n > 2048) n = 2048;  // Maximum size for reasonable memory
#else
  int n = 64;              // Default for testing
#endif

  unsigned int cutoff_size = 16;
  unsigned int cutoff_depth = 2;

  printf("Strassen Task Test (CARTS)\n");
  printf("Matrix size: %d x %d\n", n, n);
  printf("Cutoff size: %d, Cutoff depth: %d\n", cutoff_size, cutoff_depth);

  // Allocate matrices as array-of-arrays
  double **A = (double **)malloc(n * sizeof(double *));
  double **B = (double **)malloc(n * sizeof(double *));
  double **C_par = (double **)malloc(n * sizeof(double *));
  double **C_seq = (double **)malloc(n * sizeof(double *));

  if (!A || !B || !C_par || !C_seq) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  for (int i = 0; i < n; i++) {
    A[i] = (double *)malloc(n * sizeof(double));
    B[i] = (double *)malloc(n * sizeof(double));
    C_par[i] = (double *)malloc(n * sizeof(double));
    C_seq[i] = (double *)malloc(n * sizeof(double));
    if (!A[i] || !B[i] || !C_par[i] || !C_seq[i]) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
  }

  // Initialize matrices
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = (double)(rand() % 100) / 10.0;
      B[i][j] = (double)(rand() % 100) / 10.0;
      C_par[i][j] = 0.0;
      C_seq[i][j] = 0.0;
    }
  }

  printf("Running parallel Strassen with tasks...\n");
  strassen_main_par(A, B, C_par, n, cutoff_size, cutoff_depth);

  printf("Running sequential Strassen for verification...\n");
  strassen_main_seq(A, B, C_seq, n, cutoff_size);

  // Compare results
  printf("Verifying results...\n");
  double error = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double diff = C_par[i][j] - C_seq[i][j];
      error += diff * diff;
    }
  }
  error = sqrt(error / (n * n));

  printf("Verification: %s (RMS error: %.2e)\n",
         (error < 1e-4) ? "PASS" : "FAIL", error);

  // Cleanup
  for (int i = 0; i < n; i++) {
    free(A[i]);
    free(B[i]);
    free(C_par[i]);
    free(C_seq[i]);
  }
  free(A);
  free(B);
  free(C_par);
  free(C_seq);

  return (error < 1e-4) ? 0 : 1;
}

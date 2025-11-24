/* PolyBench-like 2D Jacobi (single-file, OpenMP) */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1024
#endif
#ifndef TSTEPS
#define TSTEPS 100
#endif

static void init(float **A, float **B) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (float)((i + j) % 256) * 0.001f;
      B[i][j] = 0.0f;
    }
  }
}

static void jacobi2d_step(const float **A, float **B) {
#pragma omp parallel for schedule(static)
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      B[i][j] = 0.2f * (A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
    }
  }
}

static float checksum(float **A) {
  float s = 0.0f;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      s += A[i][j];
  return s;
}

int main(void) {
  float **A = (float **)malloc(N * sizeof(float *));
  float **B = (float **)malloc(N * sizeof(float *));
  if (!A || !B) { fprintf(stderr, "alloc failed\n"); return 1; }
  
  for (int i = 0; i < N; i++) {
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
    if (!A[i] || !B[i]) { fprintf(stderr, "alloc failed\n"); return 1; }
  }
  
  init(A, B);
  for (int t = 0; t < TSTEPS; t++) {
    jacobi2d_step((const float **)A, B);
    float **tmp = A; A = B; B = tmp;
  }
  printf("checksum=%.6f\n", checksum(A));
  
  for (int i = 0; i < N; i++) {
    free(A[i]);
    free(B[i]);
  }
  free(A);
  free(B);
  return 0;
}



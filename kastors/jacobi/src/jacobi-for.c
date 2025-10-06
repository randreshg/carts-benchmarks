#include "poisson.h"

/* #pragma omp for version of SWEEP. */
void sweep(int nx, int ny, double dx, double dy, double *f_, int itold,
           int itnew, double *u_, double *unew_, int block_size) {
  int i;
  int it;
  int j;
  double (*f)[nx][ny] = (double (*)[nx][ny])f_;
  double (*u)[nx][ny] = (double (*)[nx][ny])u_;
  double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;

#pragma omp parallel shared(f, u, unew) private(i, j, it)                      \
    firstprivate(nx, ny, dx, dy, itold, itnew)
  for (it = itold + 1; it <= itnew; it++) {
    /*
       Save the current estimate.
       */
#pragma omp for
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        (*u)[i][j] = (*unew)[i][j];
      }
    }
    /*
       Compute a new estimate.
       */
#pragma omp for
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
          (*unew)[i][j] = (*f)[i][j];
        } else {
          (*unew)[i][j] =
              0.25 * ((*u)[i - 1][j] + (*u)[i][j + 1] + (*u)[i][j - 1] +
                      (*u)[i + 1][j] + (*f)[i][j] * dx * dy);
        }
      }
    }
  }
}

#include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
void sweep(int nx, int ny, double dx, double dy, double *f_, int itold,
           int itnew, double *u_, double *unew_, int block_size) {
#ifdef _OPENMP
  double (*f)[nx][ny] = (double (*)[nx][ny])f_;
  double (*u)[nx][ny] = (double (*)[nx][ny])u_;
  double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;
#endif
  if (block_size == 0)
    block_size = nx;

#pragma omp parallel shared(u_, unew_, f, nx, ny, dx, dy, itold, itnew,        \
                                block_size)
#pragma omp master
  {
    for (int it = itold + 1; it <= itnew; it++) {
      // Save the current estimate.
      for (int j = 0; j < ny; j += block_size) {
        for (int i = 0; i < nx; i += block_size) {
#pragma omp task shared(u_, unew_) firstprivate(i, j, block_size, nx, ny)      \
    depend(in : unew[i : block_size][j : block_size])                          \
    depend(out : u[i : block_size][j : block_size])
          copy_block(nx, ny, i / block_size, j / block_size, u_, unew_,
                     block_size);
        }
      }

      // Compute a new estimate.
      for (int j = 0; j < ny; j += block_size) {
        for (int i = 0; i < nx; i += block_size) {
          int xdm1 = i == 0 ? 0 : block_size;
          int xdp1 = i == nx - block_size ? 0 : block_size;
          int ydp1 = j == ny - block_size ? 0 : block_size;
          int ydm1 = j == 0 ? 0 : block_size;
#pragma omp task shared(u_, unew_)                                             \
    firstprivate(dx, dy, nx, ny, block_size, i, j, xdm1, xdp1, ydp1, ydm1)     \
    depend(out : unew[i : block_size][j : block_size])                         \
    depend(in : f[i : block_size][j : block_size],                             \
               u[i : block_size][j : block_size],                              \
               u[(i - xdm1) : block_size][j : block_size],                     \
               u[i : block_size][(j + ydp1) : block_size],                     \
               u[i : block_size][(j - ydm1) : block_size],                     \
               u[(i + xdp1) : block_size][j : block_size])
          compute_estimate(i / block_size, j / block_size, u_, unew_, f_, dx,
                           dy, nx, ny, block_size);
        }
      }
    }
  }
}

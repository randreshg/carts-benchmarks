# jacobi2d - 2D Jacobi Iterative Stencil

## Description

2D Jacobi iterative method with 5-point stencil averaging. Classic iterative solver for partial differential equations (PDEs).

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/stencils/jacobi-2d-imper/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Original Paper**: "Polybench: The polyhedral benchmark suite" (Pouchet et al.)
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." In Workshop on Polyhedral Compilation Techniques (IMPACT), 2012.

## Algorithm

5-point averaging stencil applied iteratively:

```
Input: A[N×N], TSTEPS
Output: A (modified after TSTEPS iterations)

For t = 0 to TSTEPS-1:
  for i = 1 to N-2:
    for j = 1 to N-2:
      B[i][j] = 0.2 * (A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])
  swap(A, B)
```

The computation uses two arrays (A and B) and swaps them each iteration to avoid overwriting values needed for the stencil computation.

## Problem Sizes

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 1024×1024 | Standard problem size (default) |
| **LARGE** | 2000×2000 | Large problem size |
| **EXTRALARGE** | 4000×4000 | Extra large problem size |

Default time steps: 100 iterations

## OpenMP Parallelization

Uses `#pragma omp parallel for schedule(static)` to parallelize the outer loop over rows.

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (128×128)
make small

# Build medium size (1024×1024) - default
make medium

# Build large size (2000×2000)
make large

# Build all pipeline stages (seq, metadata, parallel, concurrency)
make all
```

### Build individual stages

```bash
# Generate sequential MLIR
make seq

# Collect runtime metadata
make metadata

# Generate parallel MLIR
make parallel

# Run concurrency analysis
make concurrency

# Run optimized concurrency analysis
make concurrency-opt
```

### Clean build artifacts

```bash
make clean
```

## Use in Machine Learning

Jacobi iteration patterns appear in:
- Iterative optimization algorithms
- Fixed-point iterations in neural ODEs
- Relaxation methods in graph neural networks
- Numerical solvers used in physics-informed neural networks

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallel for
- Static functions
- 5-point stencil pattern

## References

- **Polybench Suite**: http://polybench.sourceforge.net/
- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Original Authors**: Louis-Noël Pouchet, Ohio State University
- **Jacobi Method**: Classical iterative method for solving linear systems

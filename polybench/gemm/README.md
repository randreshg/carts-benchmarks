# gemm - General Matrix Multiplication

## Description

General Matrix Multiply (GEMM): `C = alpha * A * B + beta * C`

This is the fundamental operation in linear algebra and forms the core of most deep learning computations.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/linear-algebra/blas/gemm/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Original Paper**: "Polybench: The polyhedral benchmark suite" (Pouchet et al.)
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." In Workshop on Polyhedral Compilation Techniques (IMPACT), 2012.

## Algorithm

```
Input: A[NI×NK], B[NK×NJ], C[NI×NJ], alpha, beta
Output: C (modified)

for i = 0 to NI-1:
  for j = 0 to NJ-1:
    sum = 0
    for k = 0 to NK-1:
      sum += A[i][k] * B[k][j]
    C[i][j] = alpha * sum + beta * C[i][j]
```

## Problem Sizes

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 512×512 | Standard problem size (default) |
| **LARGE** | 2000×2000 | Large problem size |
| **EXTRALARGE** | 4000×4000 | Extra large problem size |

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

GEMM is the most important kernel in deep learning:
- Fully connected (dense) layers
- Convolution (when im2col is applied)
- Attention mechanisms in transformers
- Batched matrix operations
- Gradient computations

Libraries like cuBLAS, MKL, and OpenBLAS are heavily optimized for GEMM.

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallel for
- Static functions

## References

- **Polybench Suite**: http://polybench.sourceforge.net/
- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Original Authors**: Louis-Noël Pouchet, Ohio State University
- **BLAS Standard**: http://www.netlib.org/blas/

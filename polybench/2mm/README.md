# 2mm - Two Matrix Multiplications

## Description

Performs two chained matrix multiplications: `D = alpha * A * B * C + beta * D`

This kernel computes:
1. `tmp = alpha * A * B`
2. `D = tmp * C + beta * D`

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/linear-algebra/kernels/2mm/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Original Paper**: "Polybench: The polyhedral benchmark suite" (Pouchet et al.)
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." In Workshop on Polyhedral Compilation Techniques (IMPACT), 2012.

## Algorithm

```
Input: A[NI×NK], B[NK×NJ], C[NL×NJ], D[NI×NL], alpha, beta
Output: D (modified)

tmp[NI×NJ] = 0
for i = 0 to NI-1:
  for j = 0 to NJ-1:
    for k = 0 to NK-1:
      tmp[i][j] += alpha * A[i][k] * B[k][j]

for i = 0 to NI-1:
  for j = 0 to NL-1:
    D[i][j] *= beta
    for k = 0 to NJ-1:
      D[i][j] += tmp[i][k] * C[k][j]
```

## Problem Sizes

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 1024×1024 | Standard problem size (default) |
| **LARGE** | 2000×2000 | Large problem size |
| **EXTRALARGE** | 4000×4000 | Extra large problem size |

## OpenMP Parallelization

Uses `#pragma omp parallel` with `#pragma omp for private(j, k)` for both matrix multiplication loops.

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

Two chained matrix multiplications are common in:
- Neural network layers (e.g., attention mechanisms)
- Linear algebra operations in transformers
- Multi-layer perceptron computations

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallel for
- Static functions

## References

- **Polybench Suite**: http://polybench.sourceforge.net/
- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Original Authors**: Louis-Noël Pouchet, Ohio State University

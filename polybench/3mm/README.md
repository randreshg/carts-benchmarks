# 3mm - Three Matrix Multiplications

## Description

Performs three chained matrix multiplications:
- `E = A * B`
- `F = C * D`
- `G = E * F`

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/linear-algebra/kernels/3mm/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Original Paper**: "Polybench: The polyhedral benchmark suite" (Pouchet et al.)
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." In Workshop on Polyhedral Compilation Techniques (IMPACT), 2012.

## Algorithm

```
Input: A[NI×NK], B[NK×NJ], C[NJ×NM], D[NM×NL]
Output: G[NI×NL]

E[NI×NJ] = A * B
F[NJ×NL] = C * D
G[NI×NL] = E * F
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

Uses `#pragma omp parallel private(j, k)` with multiple `#pragma omp for` regions within a single parallel region.

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

Three chained matrix multiplications test:
- Dependency chains across multiple operations
- Intermediate result management
- Memory allocation patterns for temporary matrices

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallel for
- Static functions

## References

- **Polybench Suite**: http://polybench.sourceforge.net/
- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Original Authors**: Louis-Noël Pouchet, Ohio State University

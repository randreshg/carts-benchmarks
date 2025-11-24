# SW4Lite Seismic Wave Propagation Kernels

Computational kernels extracted from SW4Lite (Seismic Waves, 4th order, Lite version), a seismic wave propagation code developed at Lawrence Livermore National Laboratory (LLNL).

## Source

**Upstream**: [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite)
**Application**: Earthquake simulation and seismic wave propagation
**Category**: Stencil computations, high-order finite differences

## Benchmarks

| Example | Description | Source File | Status |
|---------|-------------|-------------|--------|
| **rhs4sg-base** | Right-hand side assembly (baseline) | `rhs4sg_base.c` | Complete |
| **rhs4sg-revnw** | RHS assembly (revised, optimized) | `rhs4sg_revNW.c` | Complete |
| **vel4sg-base** | Velocity update phase | `vel4sg_base.c` | Complete |

### rhs4sg-base
Simplified interior RHS assembly kernel with fourth-order accurate Laplacian and Lamé coefficients. No one-sided boundary handling, focusing on interior stencil convergence.

### rhs4sg-revnw
Full-featured RHS assembly kernel with optimizations including restrict pointers and revised memory access patterns. Includes complete MLIR representation.

### vel4sg-base
Velocity update phase that integrates stress divergence divided by density, completing the SW4Lite time-stepping pair (RHS + velocity) with memory access patterns similar to the production code.

## Build Instructions

### Build all SW4Lite benchmarks
```bash
make -C sw4lite all
```

### Build individual benchmark
```bash
make -C sw4lite/rhs4sg-base all
make -C sw4lite/rhs4sg-revnw all
make -C sw4lite/vel4sg-base all
```

### Build with specific sizes
Each benchmark supports three problem sizes based on 3D grid dimensions (NX × NY × NZ):

- **small**: 10 × 10 × 10 = 1,000 elements
- **medium**: 21 × 21 × 22 ≈ 10,000 elements
- **large**: 46 × 46 × 47 ≈ 100,000 elements

```bash
make -C sw4lite/rhs4sg-base small
make -C sw4lite/rhs4sg-base medium
make -C sw4lite/rhs4sg-base large
```

### Custom grid sizes
```bash
make -C sw4lite/rhs4sg-base CFLAGS="-DNX=64 -DNY=64 -DNZ=64" all
```

## Problem Characteristics

- **Stencil pattern**: 3D fourth-order accurate (27-point stencil)
- **Operations**: Laplacian computation with Lamé material coefficients
- **Memory access**: Structured grid with spatial locality
- **Parallelization**: OpenMP parallel-for over grid points
- **Typical use**: Seismic wave equation time-stepping

## Standard CARTS Targets

```bash
make all              # Build all stages
make seq              # Sequential MLIR without OpenMP
make metadata         # Collect parallelism metadata
make parallel         # Parallel MLIR with OpenMP
make concurrency      # Run concurrency analysis
make concurrency-opt  # Run optimized concurrency analysis
make clean            # Remove build artifacts
```

## References

1. N. Anders Petersson and Björn Sjögreen. "SW4 User's Guide". LLNL-SM-741439, 2017.
2. [SW4Lite GitHub Repository](https://github.com/geodynamics/sw4lite)
3. [SW4 Project Page](https://geodynamics.org/cig/software/sw4/)

## CARTS Compatibility

- No global variables - clean parameter passing
- OpenMP parallelization with parallel-for
- Self-contained implementations
- Well-documented with upstream attribution
- Multiple problem sizes for scaling studies

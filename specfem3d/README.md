# SPECFEM3D Seismic Simulation Kernels

Computational kernels extracted from SPECFEM3D, a spectral-element method code for 3D seismic wave propagation and earthquake simulation.

## Source

**Upstream**: [SPECFEM/specfem3d](https://github.com/SPECFEM/specfem3d)
**Application**: Seismic wave propagation using spectral elements
**Category**: Spectral methods, tensor operations, elasticity

## Benchmarks

| Example | Description | Source File | Status |
|---------|-------------|-------------|--------|
| **stress** | Stress tensor update | `stress_update.c` | Complete |
| **velocity** | Velocity update | `velocity_update.c` | Complete |

### stress
Stress tensor update derived from SPECFEM3D's isotropic Hooke-law implementation. Computes six-component stress tensor updates using mixed derivatives and Lamé parameters.

### velocity
Velocity update using stress divergence and density scaling from SPECFEM3D's elastic solver. Performs divergence calculations similar to the production code's particle-velocity integration.

## Build Instructions

### Build all SPECFEM3D benchmarks
```bash
make -C specfem3d all
```

### Build individual benchmark
```bash
make -C specfem3d/stress all
make -C specfem3d/velocity all
```

### Build with specific sizes
Each benchmark supports three problem sizes based on spectral elements:

- **small**: 5×5×5 GLL points × 8 elements ≈ 1,000 degrees of freedom
- **medium**: 5×5×5 GLL points × 80 elements ≈ 10,000 degrees of freedom
- **large**: 5×5×5 GLL points × 800 elements ≈ 100,000 degrees of freedom

```bash
make -C specfem3d/stress small
make -C specfem3d/stress medium
make -C specfem3d/stress large
```

### Custom problem sizes
```bash
make -C specfem3d/stress CFLAGS="-DNGLLX=5 -DNGLLY=5 -DNGLLZ=5 -DNSPEC=100" all
```

## Problem Characteristics

- **Method**: Spectral-element (high-order Gauss-Lobatto-Legendre basis)
- **Operations**: Tensor contractions, stress/strain updates
- **Memory access**: Indirect indexing through spectral element connectivity
- **Parallelization**: OpenMP parallel-for over elements
- **Typical use**: Earthquake simulation, seismic tomography

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

1. D. Komatitsch and J. Tromp. "Introduction to the spectral element method for three-dimensional seismic wave propagation". Geophysical Journal International, 1999.
2. [SPECFEM3D GitHub Repository](https://github.com/SPECFEM/specfem3d)
3. [SPECFEM3D User Manual](https://specfem3d.readthedocs.io/)

## CARTS Compatibility

- No global variables - clean parameter passing
- OpenMP parallelization with parallel-for
- Self-contained implementations
- Well-documented with upstream attribution
- Multiple problem sizes for scaling studies

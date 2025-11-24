# SeisSol High-Order Discontinuous Galerkin Kernels

Computational kernels extracted from SeisSol, a high-order discontinuous Galerkin (DG) code for seismic wave propagation and earthquake dynamic rupture simulation.

## Source

**Upstream**: [SeisSol/SeisSol](https://github.com/SeisSol/SeisSol)
**Application**: Seismic wave propagation using ADER-DG methods
**Category**: Discontinuous Galerkin, tensor contractions, dense linear algebra

## Benchmarks

| Example | Description | Source File | Status |
|---------|-------------|-------------|--------|
| **volume-integral** | Element-local volume integral | `volume_integral.c` | Complete |

### volume-integral
Element-local ADER-DG volume integral computation extracted from SeisSol-generated kernels. Performs dense tensor contractions and matrix-matrix multiplies typical of high-order DG solvers.

## Build Instructions

### Build all SeisSol benchmarks
```bash
make -C seissol all
```

### Build individual benchmark
```bash
make -C seissol/volume-integral all
```

### Build with specific sizes
Each benchmark supports three problem sizes based on number of elements:

- **small**: 1,000 elements
- **medium**: 10,000 elements
- **large**: 100,000 elements

```bash
make -C seissol/volume-integral small
make -C seissol/volume-integral medium
make -C seissol/volume-integral large
```

### Custom problem sizes
```bash
make -C seissol/volume-integral CFLAGS="-DNELEM=5000" all
```

## Problem Characteristics

- **Method**: ADER-DG (Arbitrary high-order DERivatives Discontinuous Galerkin)
- **Operations**: Dense tensor contractions, batched matrix multiplies
- **Memory access**: Element-local operations with high computational intensity
- **Parallelization**: OpenMP parallel-for over elements
- **Typical use**: High-resolution earthquake simulation, rupture dynamics

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

1. M. Dumbser and M. Käser. "An arbitrary high-order discontinuous Galerkin method for elastic waves on unstructured meshes". GJI, 2006.
2. A. Breuer et al. "Sustained Petascale Performance of Seismic Simulations with SeisSol on SuperMUC". ISC, 2014.
3. [SeisSol GitHub Repository](https://github.com/SeisSol/SeisSol)
4. [SeisSol Documentation](https://seissol.readthedocs.io/)

## CARTS Compatibility

- No global variables - clean parameter passing
- OpenMP parallelization with parallel-for
- Self-contained implementations
- Well-documented with upstream attribution
- Multiple problem sizes for scaling studies

# Volume Integral Benchmark

This benchmark implements a simplified volume integral computation from the SeisSol earthquake simulation code. It demonstrates parallelization of element-wise matrix operations using CARTS.

## Description

The volume integral computes flux contributions for each element in a mesh:

1. **Gradient computation**: For each element, compute `buffer[q] = sum_b(gradMatrix[q][b] * dofs[elem][b])`
2. **Flux accumulation**: For each basis function, compute `fluxOut[elem][b] = sum_q(fluxMatrix[q][b] * buffer[q])`

The outer loop over elements (`elem`) is parallelized with OpenMP.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ELEMENTS` | 64 | Number of mesh elements |
| `N_BASIS` | 30 | Number of basis functions |
| `N_QUAD` | 36 | Number of quadrature points |

## Building

```bash
# Build with default parameters
make all

# Build with custom parameters
make all N_ELEMENTS=128 N_BASIS=20 N_QUAD=25

# Clean build artifacts
make clean
```

## Running

```bash
# Run the ARTS-parallelized version
./volume_integral_arts
```

Expected output:
```
seissol_volume_integral checksum=<value>
```

## Source Structure

- `volume_integral.c` - Main source file with OpenMP-parallelized volume integral
- `arts.cfg` - ARTS runtime configuration (threads, ports, etc.)
- `Makefile` - Build configuration using `../common/carts-example.mk`
- [CARTS pipeline guide](/opt/carts/docs/agents.md) - Debugging and pipeline reference

## Key Implementation Pattern

```c
#pragma omp parallel for schedule(static)
for (int elem = 0; elem < N_ELEMENTS; ++elem) {
    float buffer[N_QUAD];  // Thread-local temporary storage

    // Compute gradient at quadrature points
    for (int q = 0; q < N_QUAD; ++q) {
        float val = 0.0f;
        for (int b = 0; b < N_BASIS; ++b) {
            val += gradMatrix[q][b] * dofs[elem][b];
        }
        buffer[q] = val;
    }

    // Accumulate flux contributions
    for (int b = 0; b < N_BASIS; ++b) {
        float acc = 0.0f;
        for (int q = 0; q < N_QUAD; ++q) {
            acc += fluxMatrix[q][b] * buffer[q];
        }
        fluxOut[elem][b] = acc;
    }
}
```

## Debugging

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps and pipeline analysis commands.

# SW4Lite RHS4SG (Base) Benchmark

Simplified interior right-hand-side assembly kernel from SW4Lite, focusing on fourth-order accurate Laplacian computations with Lamé coefficients.

## Origin

- **Source**: [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite) (LLNL)
- **Pattern**: 3D fourth-order accurate Laplacian with Lamé coefficients for seismic wave equations
- **Purpose**: Baseline interior stencil without one-sided boundary handling

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NX` | 48 | Grid size in X direction |
| `NY` | 48 | Grid size in Y direction |
| `NZ` | 48 | Grid size in Z direction |
| `POINTS_PER_DIR` | 4 | Boundary padding points |
| `COMP` | 3 | Number of components (fixed) |

## Building

```bash
# Build with default parameters
make all

# Build with custom parameters
make all NX=64 NY=64 NZ=64

# Clean build artifacts
make clean
```

## Running

```bash
# Run the ARTS-parallelized version
./rhs4sg_base_arts
```

Expected output:
```
sw4lite_rhs4sg_base checksum=0.019759
```

## Source Structure

- `rhs4sg_base.c` - Main source file with OpenMP-parallelized RHS kernel
- `arts.cfg` - ARTS runtime configuration
- `Makefile` - Build configuration
- [CARTS pipeline guide](/opt/carts/docs/agents.md) - Debugging and pipeline reference

## Characteristics

- **Stencil**: 3D 27-point (4th-order accurate)
- **Grid**: Uniform Cartesian mesh
- **Operations**: Laplacian + Lamé material coefficients
- **Parallelization**: OpenMP parallel-for over grid points

## Key Implementation Pattern

```c
#pragma omp parallel for schedule(static)
for (int k = POINTS_PER_DIR; k < NZ - POINTS_PER_DIR; ++k) {
  for (int j = POINTS_PER_DIR; j < NY - POINTS_PER_DIR; ++j) {
    for (int i = POINTS_PER_DIR; i < NX - POINTS_PER_DIR; ++i) {
      // Compute Laplacian using 4th-order stencil weights
      float lap = 0.0f;
      for (int offset = -2; offset <= 2; ++offset) {
        lap += w[offset + 2] * (u[c][i + offset][j][k] +
                                u[c][i][j + offset][k] +
                                u[c][i][j][k + offset]);
      }
      // Apply Lamé coefficients
      rhs[c][i][j][k] = mu_c * lap * inv_h2 + ...;
    }
  }
}
```

## CARTS Compatibility Notes

### 1. Removed `collapse(2)` clause
```c
// Original:
#pragma omp parallel for collapse(2) schedule(static)

// Modified:
#pragma omp parallel for schedule(static)
```

### 2. Unrolled component loop to eliminate if-else chain

The original code used a conditional inside the loop to select the divergence direction:
```c
// Original:
for (int c = 0; c < COMP; ++c) {
  float lap = 0.0f;
  for (int offset = -2; offset <= 2; ++offset) {
    lap += w[offset + 2] * (u[c][i + offset][j][k] + ...);
  }
  float div_term = 0.0f;
  if (c == 0)      div_term = u[c][i + 1][j][k] - u[c][i - 1][j][k];  // x-dir
  else if (c == 1) div_term = u[c][i][j + 1][k] - u[c][i][j - 1][k];  // y-dir
  else             div_term = u[c][i][j][k + 1] - u[c][i][j][k - 1];  // z-dir
  rhs[c][i][j][k] = mu_c * lap * inv_h2 + (la_c + mu_c) * div_term * (0.5f / h);
}
```

The modified code manually unrolls the 3 components, computing the same values:
```c
// Modified (semantically equivalent):
// Component 0: lap uses u[0], div_term uses x-direction (i±1)
// Component 1: lap uses u[1], div_term uses y-direction (j±1)
// Component 2: lap uses u[2], div_term uses z-direction (k±1)
```

This is necessary because `if-else` chains create multiple basic blocks in MLIR, which violates `scf.for`'s single-block requirement.

### 3. Removed early return
```c
// Original:
if (!u || !mu || !lambda || !rhs) {
  fprintf(stderr, "allocation failure\n");
  return 1;
}

// Modified: removed (assumes allocations succeed in benchmark context)
```

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps.

## See Also

- **rhs4sg-revnw**: Full-featured optimized version with restrict pointers
- **vel4sg-base**: Complementary velocity update kernel

# SW4Lite Velocity Update (Base) Benchmark

Velocity update kernel from SW4Lite that updates velocity components from stress tensor divergence.

## Origin

- **Source**: [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite) (LLNL)
- **Pattern**: Update velocity components (vx, vy, vz) from stress divergence divided by density
- **Purpose**: Completes the SW4Lite coverage - pairs with rhs4sg-base for full elastic wave update

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NX` | 48 | Grid size in X direction |
| `NY` | 48 | Grid size in Y direction |
| `NZ` | 48 | Grid size in Z direction |
| `DT` | 0.0005f | Time step size |

## Building

```bash
# Build with default parameters
make all

# Build with custom parameters
make all NX=64 NY=64 NZ=64 DT=0.00025f

# Clean build artifacts
make clean
```

## Running

```bash
# Run the ARTS-parallelized version
./vel4sg_base_arts
```

Expected output:
```
sw4lite_vel4sg_base checksum=0.000000
```

## Source Structure

- `vel4sg_base.c` - Main source file with OpenMP-parallelized velocity update kernel
- `arts.cfg` - ARTS runtime configuration
- `Makefile` - Build configuration
- [CARTS pipeline guide](/opt/carts/docs/agents.md) - Debugging and pipeline reference

## Key Implementation Pattern

```c
#pragma omp parallel for schedule(static)
for (int k = 2; k < NZ - 2; ++k) {
  for (int j = 2; j < NY - 2; ++j) {
    for (int i = 2; i < NX - 2; ++i) {
      const float inv_rho = 1.0f / rho[i][j][k];

      // Compute stress divergence for each velocity component
      const float div_vx = derivative_x(sxx,...) + derivative_y(sxy,...) + derivative_z(sxz,...);
      const float div_vy = derivative_x(sxy,...) + derivative_y(syy,...) + derivative_z(syz,...);
      const float div_vz = derivative_x(sxz,...) + derivative_y(syz,...) + derivative_z(szz,...);

      // Update velocities
      vx[i][j][k] += DT * inv_rho * div_vx;
      vy[i][j][k] += DT * inv_rho * div_vy;
      vz[i][j][k] += DT * inv_rho * div_vz;
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

### 2. Replaced switch-based derivative with separate functions

**Original:**
```c
static inline float derivative(const float ***arr, int i, int j, int k, int dir) {
  switch (dir) {
  case 0:  return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
  case 1:  return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
  default: return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
  }
}
```

**Modified (semantically equivalent):**
```c
static inline float derivative_x(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
}
static inline float derivative_y(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
}
static inline float derivative_z(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
}
```

### 3. Removed early return
```c
// Original:
if (!vx || !vy || ...) { return 1; }

// Modified: removed (assumes allocations succeed in benchmark context)
```

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps.

## See Also

- **rhs4sg-base**: Complementary RHS assembly kernel (computes stress contributions)

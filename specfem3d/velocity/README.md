# SPECFEM3D Velocity Update Benchmark

This benchmark implements a velocity field update routine inspired by [SPECFEM3D](https://github.com/SPECFEM/specfem3d), a spectral-element code for seismic wave propagation.

## Description

The velocity update computes new velocity components based on stress tensor gradients:

- **vx**: Updated using gradients of sxx, sxy, sxz
- **vy**: Updated using gradients of sxy, syy, syz
- **vz**: Updated using gradients of sxz, syz, szz

Each velocity component is scaled by the inverse density (1/rho) and time step (DT).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NX` | 48 | Grid size in X direction |
| `NY` | 48 | Grid size in Y direction |
| `NZ` | 48 | Grid size in Z direction |
| `DT` | 0.001f | Time step size |

## Building

```bash
# Build with default parameters
make all

# Build with custom parameters
make all NX=64 NY=64 NZ=64 DT=5e-4

# Clean build artifacts
make clean
```

## Running

```bash
# Run the ARTS-parallelized version
./velocity_update_arts
```

Expected output:
```
specfem3d_velocity checksum=-0.000000
```

## Source Structure

- `velocity_update.c` - Main source file with OpenMP-parallelized velocity update kernel
- `arts.cfg` - ARTS runtime configuration (threads, ports, etc.)
- `Makefile` - Build configuration using `../common/carts-example.mk`
- [CARTS pipeline guide](/opt/carts/docs/agents.md) - Debugging and pipeline reference

## Key Implementation Pattern

```c
#pragma omp parallel for schedule(static)
for (int k = 1; k < NZ - 1; ++k) {
  for (int j = 1; j < NY - 1; ++j) {
    for (int i = 1; i < NX - 1; ++i) {
      const float inv_rho = 1.0f / rho[i][j][k];

      // Compute velocity increments from stress gradients
      const float dvx = diff_x(sxx,...) + diff_y(sxy,...) + diff_z(sxz,...);
      const float dvy = diff_x(sxy,...) + diff_y(syy,...) + diff_z(syz,...);
      const float dvz = diff_x(sxz,...) + diff_y(syz,...) + diff_z(szz,...);

      // Update velocities
      vx[i][j][k] += DT * inv_rho * dvx;
      vy[i][j][k] += DT * inv_rho * dvy;
      vz[i][j][k] += DT * inv_rho * dvz;
    }
  }
}
```

## CARTS Compatibility Notes

1. **No switch statements** - Replaced `diff()` with separate `diff_x/y/z` functions
2. **No collapse clause** - Removed `collapse(2)` from OpenMP pragma
3. **No early returns** - Removed allocation failure check

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps and workarounds.

# SPECFEM3D Stress Update Benchmark

This benchmark implements a stress tensor update routine inspired by [SPECFEM3D](https://github.com/SPECFEM/specfem3d), a spectral-element code for seismic wave propagation.

## Description

The stress update computes the six independent stress tensor components based on velocity gradients using Hooke's law for isotropic elastic media:

1. **Normal stresses** (sxx, syy, szz): Updated using velocity derivatives and Lame parameters (mu, lambda)
2. **Shear stresses** (sxy, sxz, syz): Updated using cross-derivatives of velocity components

The computation uses a stencil pattern with central finite differences for spatial derivatives.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NX` | 40 | Grid size in X direction |
| `NY` | 40 | Grid size in Y direction |
| `NZ` | 40 | Grid size in Z direction |
| `DT` | 0.001f | Time step size |

## Building

```bash
# Build with default parameters
make all

# Build with custom parameters
make all NX=64 NY=48 NZ=48 DT=5e-4

# Clean build artifacts
make clean
```

## Running

```bash
# Run the ARTS-parallelized version
./stress_update_arts
```

Expected output:
```
specfem3d_stress checksum=0.000558
```

## Source Structure

- `stress_update.c` - Main source file with OpenMP-parallelized stress update kernel
- `arts.cfg` - ARTS runtime configuration (threads, ports, etc.)
- `Makefile` - Build configuration using `../common/carts-example.mk`
- [CARTS pipeline guide](/opt/carts/docs/agents.md) - Debugging and pipeline reference

## Key Implementation Pattern

```c
#pragma omp parallel for schedule(static)
for (int k = 2; k < NZ - 2; ++k) {
  for (int j = 2; j < NY - 2; ++j) {
    for (int i = 2; i < NX - 2; ++i) {
      // Compute velocity gradients using finite differences
      const float dvx_dx = derivative_x(vx, i, j, k);
      const float dvy_dy = derivative_y(vy, i, j, k);
      const float dvz_dz = derivative_z(vz, i, j, k);

      // Update normal stresses (Hooke's law)
      const float trace = dvx_dx + dvy_dy + dvz_dz;
      sxx[i][j][k] += DT * (2*mu*dvx_dx + lambda*trace);
      syy[i][j][k] += DT * (2*mu*dvy_dy + lambda*trace);
      szz[i][j][k] += DT * (2*mu*dvz_dz + lambda*trace);

      // Update shear stresses
      sxy[i][j][k] += DT * mu * (derivative_y(vx,...) + derivative_x(vy,...));
      sxz[i][j][k] += DT * mu * (derivative_z(vx,...) + derivative_x(vz,...));
      syz[i][j][k] += DT * mu * (derivative_z(vy,...) + derivative_y(vz,...));
    }
  }
}
```

## CARTS Compatibility Notes

1. **No switch statements** - Replaced with separate derivative functions
2. **No collapse clause** - Removed `collapse(2)` from OpenMP pragma
3. **No early returns** - Removed allocation failure check

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps and workarounds.

## Debugging

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps and pipeline analysis commands.

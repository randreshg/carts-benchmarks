# SW4Lite Seismic Wave Propagation Kernels

Computational kernels extracted from SW4Lite (Seismic Waves, 4th order, Lite version), a seismic wave propagation code developed at Lawrence Livermore National Laboratory (LLNL).

## Source

**Upstream**: [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite)
**Application**: Earthquake simulation and seismic wave propagation
**Category**: Stencil computations, high-order finite differences

## How SW4Lite Time-Stepping Works

SW4Lite solves the elastic wave equation using a staggered-grid finite difference method. Each time step consists of two phases that work together:

```
Time Step Loop:
  1. RHS Assembly (rhs4sg)  →  Compute stress contributions from displacement
  2. Velocity Update (vel4sg)  →  Update velocities from stress divergence
  (repeat)
```

### Phase 1: RHS Assembly (`rhs4sg-base`)

Computes the right-hand side of the wave equation using displacement field and material properties:

```
For each grid point (i,j,k) and component c:
  1. Compute 4th-order Laplacian of displacement:
     lap = Σ w[offset] * (u[i±offset,j,k] + u[i,j±offset,k] + u[i,j,k±offset])
     (uses 5-point stencil with weights: -1/12, 2/3, 0, -2/3, 1/12)

  2. Compute divergence term (direction depends on component):
     c=0: div = u[i+1] - u[i-1]  (x-direction)
     c=1: div = u[j+1] - u[j-1]  (y-direction)
     c=2: div = u[k+1] - u[k-1]  (z-direction)

  3. Combine with Lamé coefficients:
     rhs[c] = μ * lap + (λ + μ) * div
```

**Inputs**: displacement `u[3][NX][NY][NZ]`, material properties `mu`, `lambda`
**Outputs**: right-hand side `rhs[3][NX][NY][NZ]`

### Phase 2: Velocity Update (`vel4sg-base`)

Updates velocity components from stress tensor divergence:

```
For each grid point (i,j,k):
  1. Compute stress divergence for each velocity component:
     div_vx = ∂sxx/∂x + ∂sxy/∂y + ∂sxz/∂z
     div_vy = ∂sxy/∂x + ∂syy/∂y + ∂syz/∂z
     div_vz = ∂sxz/∂x + ∂syz/∂y + ∂szz/∂z
     (uses 2nd-order central differences: 0.5 * (arr[i+1] - arr[i-1]))

  2. Update velocities:
     vx += dt * div_vx / ρ
     vy += dt * div_vy / ρ
     vz += dt * div_vz / ρ
```

**Inputs**: stress tensors `sxx,syy,szz,sxy,sxz,syz`, density `rho`
**Outputs**: velocities `vx,vy,vz`

## Key Differences Between Kernels

| Aspect | rhs4sg-base | vel4sg-base |
|--------|-------------|-------------|
| **Purpose** | Compute stress from displacement | Update velocity from stress |
| **Stencil Order** | 4th-order (5-point) | 2nd-order (2-point) |
| **Stencil Weights** | [-1/12, 2/3, 0, -2/3, 1/12] | [0.5, -0.5] |
| **Input Arrays** | u (displacement), mu, lambda | stress tensors, rho |
| **Output Arrays** | rhs | vx, vy, vz |
| **Components** | 3 (with direction-dependent divergence) | 3 (symmetric treatment) |

## Benchmarks

| Example | Description | Status |
|---------|-------------|--------|
| **rhs4sg-base** | RHS assembly with 4th-order Laplacian | Working |
| **vel4sg-base** | Velocity update from stress divergence | Working |

## Build Instructions

### Build all SW4Lite benchmarks
```bash
make -C sw4lite all
```

### Build individual benchmark
```bash
make -C sw4lite/rhs4sg-base all
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

## CARTS Compatibility Notes

Both kernels required the same modifications for CARTS:

1. **No switch statements** - Replaced with separate `derivative_x/y/z` functions
2. **No collapse clause** - Removed `collapse(2)` from OpenMP pragma
3. **No early returns** - Removed allocation failure checks

See the [CARTS pipeline guide](/opt/carts/docs/agents.md) for detailed debugging steps.

## References

1. N. Anders Petersson and Björn Sjögreen. "SW4 User's Guide". LLNL-SM-741439, 2017.
2. [SW4Lite GitHub Repository](https://github.com/geodynamics/sw4lite)
3. [SW4 Project Page](https://geodynamics.org/cig/software/sw4/)

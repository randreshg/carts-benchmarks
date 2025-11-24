# convolution-2d - 2D Convolution (3×3 Kernel)

## Description

2D convolution with 3×3 filter kernel. Fundamental operation in Convolutional Neural Networks (CNNs).

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/stencils/convolution-2d/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." IMPACT, 2012.

## Algorithm

3×3 convolution with learned filter weights:
```
B[i][j] = Σ(m=-1 to 1) Σ(n=-1 to 1) weight[m][n] * A[i+m][j+n]
```

## Problem Sizes

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 1024×1024 | Standard problem size (default) |
| **LARGE** | 4096×4096 | Large problem size |
| **EXTRALARGE** | 8192×8192 | Extra large problem size |

## OpenMP Parallelization

Uses `#pragma omp parallel for private(j) collapse(2) schedule(static)`

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

2D convolution is the fundamental operation in CNNs:
- Feature extraction in convolutional layers
- Most common filter size in practice (3×3)
- Used in: ResNet, VGG, MobileNet, etc.

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP with collapse(2)
- 9-point stencil pattern

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **CNNs**: LeCun, Y., et al. "Gradient-based learning applied to document recognition." IEEE, 1998.
- **Modern Usage**: He, K., et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.

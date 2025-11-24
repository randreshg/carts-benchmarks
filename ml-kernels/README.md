# ML Kernels for CARTS

Machine Learning and Neural Network computational kernels extracted and adapted for CARTS testing.

## Overview

This directory contains fundamental ML/AI kernels extracted from production frameworks (Darknet) and implemented for CARTS compatibility. All kernels:

- **No global variables** - Clean parameter passing
- **OpenMP parallelization** - Using `#pragma omp parallel for`
- **Self-contained** - Minimal dependencies (stdlib, math.h)
- **Configurable sizes** - Via compile-time macros
- **Include validation** - Self-checking test cases

## Kernels

### 1. Batch Normalization (`batchnorm/`)

**Source**: Extracted from [Darknet](https://github.com/pjreddie/darknet)

**Description**: Normalizes activations across the batch dimension. Essential in modern CNNs.

**Algorithm**:
```
For each channel c:
  1. Compute mean: μ_c = mean(x[b,c,h,w] for all b,h,w)
  2. Compute variance: σ²_c = var(x[b,c,h,w] for all b,h,w)
  3. Normalize: x̂[b,c,h,w] = (x[b,c,h,w] - μ_c) / sqrt(σ²_c + ε)
  4. Scale and shift: y[b,c,h,w] = γ_c * x̂[b,c,h,w] + β_c
```

**Functions**:
- `mean_cpu()` - Compute channel-wise mean
- `variance_cpu()` - Compute channel-wise variance
- `normalize_cpu()` - Normalize using mean and variance
- `scale_bias_cpu()` - Apply learned scale
- `add_bias_cpu()` - Apply learned bias
- `batchnorm_forward()` - Complete forward pass

**Memory Layout**: NCHW (batch, channels, height, width)

**Build**:
```bash
cd batchnorm/
make              # Standard size
make mini         # 2x8x8x8
make small        # 4x16x16x16
make standard     # 4x64x32x32
make large        # 8x128x64x64
```

**Usage**:
```c
batchnorm_forward(input, output, scales, biases,
                  batch, channels, height, width,
                  mean, variance);
```

---

### 2. Pooling Operations (`pooling/`)

**Source**: Extracted from [Darknet](https://github.com/pjreddie/darknet)

**Description**: Downsampling operations for CNNs. Reduces spatial dimensions.

**Kernels Implemented**:

#### Max Pooling
Takes maximum value in each pooling window.
```
output[b,c,i,j] = max(input[b,c,i*stride+m,j*stride+n]
                      for m,n in [0, pool_size))
```

#### Average Pooling
Takes average of values in each pooling window.
```
output[b,c,i,j] = mean(input[b,c,i*stride+m,j*stride+n]
                       for m,n in [0, pool_size))
```

#### Global Average Pooling
Averages across entire spatial dimensions (used before classification).
```
output[b,c] = mean(input[b,c,h,w] for all h,w)
```

**Functions**:
- `maxpool_forward()` - Max pooling with configurable window and stride
- `avgpool_forward()` - Average pooling with configurable window and stride
- `global_avgpool()` - Global average pooling (spatial → vector)

**Typical Configuration**: 2×2 pool, stride 2 (halves spatial dimensions)

**Build**:
```bash
cd pooling/
make              # Standard size
make mini         # 2x8x16x16
make small        # 4x16x32x32
make standard     # 4x64x64x64
make large        # 8x128x128x128
```

**Usage**:
```c
// Max pooling
maxpool_forward(input, output, batch, channels,
                in_height, in_width, pool_size, stride, padding);

// Average pooling
avgpool_forward(input, output, batch, channels,
                in_height, in_width, pool_size, stride, padding);

// Global average pooling
global_avgpool(input, output, batch, channels, height, width);
```

---

### 3. Layer Normalization (`layernorm/`)

**Source**: Inspired by [PyTorch LayerNorm](https://github.com/pytorch/pytorch)

**Description**: Normalizes each token across the hidden dimension, commonly used in transformers and large language models.

**Algorithm**:
```
for b in batch:
  mean = avg(x[b, :])
  var = avg((x[b, :] - mean)^2)
  y[b, :] = ((x[b, :] - mean) / sqrt(var + eps)) * gamma[:] + beta[:]
```

**Features**:
- OpenMP parallelism across batch dimension
- Configurable `BATCH` and `HIDDEN` via `CFLAGS`
- Affine weights (`gamma`, `beta`) stored per hidden unit

**Build**:
```bash
cd layernorm/
make           # Standard size (16x1024)
make mini      # 4 x 256 (fast smoke)
make large     # 64 x 4096 (stress)
```

**Use cases**: Validate CARTS metadata for reduction-heavy kernels and act as a bridge to more complex transformer blocks.

---

### 3. Activation Functions (`activations/`)

**Source**: ReLU from Darknet, GELU implemented, Softmax from llama2-transformer

**Description**: Non-linear activation functions used throughout neural networks.

**Functions Implemented**:

#### ReLU (Rectified Linear Unit)
```c
void activate_relu(float *x, int n)
f(x) = max(0, x)
```
- Most common activation in CNNs
- Simple, fast, effective

#### Leaky ReLU
```c
void activate_leaky(float *x, int n)
f(x) = max(0.1x, x)
```
- Allows small negative values
- Helps with "dying ReLU" problem

#### ReLU6
```c
void activate_relu6(float *x, int n)
f(x) = min(max(0, x), 6)
```
- Used in mobile networks (MobileNet)
- Limits maximum value to 6

#### GELU (Gaussian Error Linear Unit)
```c
void activate_gelu(float *x, int n)
f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```
- Used in transformers (BERT, GPT-2, GPT-3)
- Smoother than ReLU, better gradient flow

#### GELU Fast (Approximation)
```c
void activate_gelu_fast(float *x, int n)
f(x) = x * σ(1.702 * x)
```
- Faster approximation of GELU
- Good for inference

#### Sigmoid
```c
void activate_sigmoid(float *x, int n)
f(x) = 1 / (1 + exp(-x))
```
- Maps to (0, 1) range
- Used in binary classification

#### Tanh
```c
void activate_tanh(float *x, int n)
f(x) = tanh(x)
```
- Maps to (-1, 1) range
- Zero-centered

#### Softmax
```c
void softmax(float *x, int n)
f(x_i) = exp(x_i) / Σ exp(x_j)
```
- Converts scores to probabilities
- Used in classification output
- Numerically stable implementation

**Build**:
```bash
cd activations/
make              # Standard size (1M elements)
make mini         # 1K elements
make small        # 64K elements
make standard     # 1M elements
make large        # 16M elements
```

**Usage**:
```c
// In-place activation
activate_relu(x, size);
activate_gelu(x, size);
softmax(x, size);  // Note: sum(x) will be 1.0 after this
```

---

## Building All Kernels

### Standard Build (gcc)
```bash
# Build all with standard sizes
cd ml-kernels/
for dir in batchnorm pooling activations; do
  cd $dir && make && cd ..
done
```

### Build with OpenMP
```bash
# Add OpenMP flag for parallelization
cd ml-kernels/batchnorm/
gcc -O2 -fopenmp -lm batchnorm.c -o batchnorm -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32
```

### Build with CARTS
```bash
cd /Users/randreshg/Documents/carts

# Batch normalization
./tools/carts run cgeist \
  external/carts-benchmarks/ml-kernels/batchnorm/batchnorm.c \
  -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32 \
  -fopenmp -O2 -lm

# Pooling
./tools/carts run cgeist \
  external/carts-benchmarks/ml-kernels/pooling/pooling.c \
  -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=64 -DWIDTH=64 \
  -DPOOL_SIZE=2 -DSTRIDE=2 \
  -fopenmp -O2 -lm

# Activations
./tools/carts run cgeist \
  external/carts-benchmarks/ml-kernels/activations/activations.c \
  -DSIZE=1048576 \
  -fopenmp -O2 -lm
```

---

## CARTS Testing Focus

### Batch Normalization
- **Reduction operations**: Mean and variance across batch/spatial dimensions
- **Broadcast operations**: Scale and bias applied per-channel
- **Data dependencies**: Normalize depends on statistics computation
- **Memory access patterns**: Gather (for stats) then scatter (for normalize)

### Pooling
- **Irregular memory access**: Non-contiguous reads within pooling windows
- **Reduction operations**: Max or average over window
- **Spatial dependencies**: Output depends on overlapping input regions
- **Comparison operations**: Max pooling has conditional logic

### Activations
- **Element-wise operations**: Simple parallel operations
- **Transcendental functions**: GELU uses tanh, exp (tests function call analysis)
- **Conditional operations**: ReLU has branches
- **Numerical stability**: Softmax tests max subtraction pattern

---

## Validation

All kernels include self-validation:

### Batch Normalization
- Checks that statistics are computed correctly
- Prints sample mean and variance values
- Validates output range

### Pooling
- Validates max_pool >= avg_pool (element-wise)
- Checks spatial dimension reduction
- Tests boundary handling

### Activations
- Validates output ranges (e.g., ReLU >= 0, sigmoid in (0,1))
- Softmax sum validation (should equal 1.0)
- Compares exact vs approximation (GELU vs GELU fast)

---

## File Structure

```
ml-kernels/
├── README.md                      # This file
├── batchnorm/
│   ├── batchnorm.c               # Batch normalization kernel
│   └── Makefile                  # Build system
├── pooling/
│   ├── pooling.c                 # Max, average, global pooling
│   └── Makefile                  # Build system
└── activations/
    ├── activations.c             # 8 activation functions
    └── Makefile                  # Build system
```

---

## Integration with Existing carts benchmarkss

These ML kernels complement existing carts benchmarkss:

| ML Kernel | Related carts benchmarks | Key Difference |
|-----------|------------------------|----------------|
| Batch Norm | - | New: channel-wise normalization |
| Max/Avg Pool | Polybench stencils | Reduction vs weighted sum |
| ReLU | - | New: simple activation |
| GELU | - | New: transformer activation |
| Softmax | llama2 transformer | Extracted standalone version |

---

## Problem Size Guidelines

### For Development/Testing
- **MINI**: Fast compilation and execution (~seconds)
- **SMALL**: Quick testing (~tens of seconds)

### For Benchmarking
- **STANDARD**: Realistic workload sizes (~minutes)
- **LARGE**: Stress testing (~tens of minutes)

### Memory Requirements

| Kernel | MINI | SMALL | STANDARD | LARGE |
|--------|------|-------|----------|-------|
| **Batch Norm** | ~8 KB | ~256 KB | ~32 MB | ~256 MB |
| **Pooling** | ~16 KB | ~256 KB | ~64 MB | ~512 MB |
| **Activations** | ~4 KB | ~256 KB | ~4 MB | ~64 MB |

---

## Performance Characteristics

### Batch Normalization
- **Compute**: Mean and variance are reductions (O(BCHW))
- **Memory**: 2 passes through data (read for stats, read/write for normalize)
- **Parallelization**: Good (channel-wise or batch-wise)

### Pooling
- **Compute**: Simple max/average (O(BCHW × pool_size²))
- **Memory**: Strided reads (cache-friendly if stride ≈ pool_size)
- **Parallelization**: Excellent (independent output positions)

### Activations
- **Compute**: Element-wise (O(N))
- **Memory**: Streaming (in-place modification)
- **Parallelization**: Perfect (embarrassingly parallel)
- **Note**: GELU slower than ReLU due to transcendental functions

---

## References

- **Darknet**: https://github.com/pjreddie/darknet
- **Batch Normalization**: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
- **Pooling**: Standard operation in CNNs since LeNet-5

---

**Created**: November 11, 2025
**Status**: Ready for CARTS testing
**Total Kernels**: 3 files, 15+ functions
**Lines of Code**: ~1300 lines

# Activation Functions - Neural Network Non-linearity Operations

## Description

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. This benchmark implements eight common activation functions: ReLU, Leaky ReLU, ReLU6, GELU (standard and fast), Sigmoid, Tanh, and Softmax. These element-wise operations are fundamental building blocks in all modern deep learning architectures.

## Algorithm

Each activation function transforms input values according to its specific mathematical formula:

```
ReLU:        f(x) = max(0, x)
Leaky ReLU:  f(x) = max(0.1x, x)
ReLU6:       f(x) = min(max(0, x), 6)
GELU:        f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
GELU Fast:   f(x) = x * σ(1.702 * x)
Sigmoid:     f(x) = 1 / (1 + exp(-x))
Tanh:        f(x) = tanh(x)
Softmax:     f(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **MINI** | SIZE=1024 | Minimal size for quick testing |
| **SMALL** | SIZE=65536 | Small problem size (64K elements) |
| **MEDIUM** | SIZE=1048576 | Medium problem size - default (1M elements) |
| **STANDARD** | SIZE=1048576 | Standard problem size (same as medium) |
| **LARGE** | SIZE=16777216, NREPS=50 | Large problem size (16M elements, tuned for full thread sweeps) |

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size
make small

# Build medium size - default
make medium

# Build large size
make large

# Build all pipeline stages
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

Activation functions are applied after linear layers in neural networks:

- **ReLU**: Most common in CNNs (ResNet, VGG); fast and effective; suffers from "dying ReLU" problem
- **Leaky ReLU**: Addresses dying ReLU by allowing small negative gradients; used in GANs
- **ReLU6**: Mobile networks (MobileNet, MobileNetV2); bounds output for quantization
- **GELU**: Transformers (BERT, GPT-2, GPT-3, GPT-4); smoother gradients than ReLU
- **GELU Fast**: Faster approximation for inference; common in optimized transformer deployments
- **Sigmoid**: Binary classification output layers; gates in LSTM/GRU
- **Tanh**: LSTMs, RNNs; zero-centered unlike sigmoid
- **Softmax**: Multi-class classification output; attention mechanisms in transformers

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallelization
- Element-wise operations

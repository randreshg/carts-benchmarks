# Llama2 Transformer - Decoder-Only Language Model

## Description

Simplified implementation of the Llama2 transformer architecture for language modeling. This is a decoder-only transformer with multi-head attention, RMSNorm, and SwiGLU feedforward networks.

## Original Source

**Repository**: [llama2.c by Andrej Karpathy](https://github.com/karpathy/llama2.c)
**License**: MIT License
**Author**: Andrej Karpathy

**llama2.c Project**:
- **Repository**: https://github.com/karpathy/llama2.c
- **Purpose**: Simple, pure C/CUDA implementation of Llama2 inference
- **Philosophy**: "I want to have a minimal, educational, and hackable implementation"

## Algorithm

### Transformer Architecture

```
For each layer l in [0, N_LAYERS):
  1. Attention:
     - RMSNorm(x)
     - Q = x @ Wq, K = x @ Wk, V = x @ Wv
     - Attention scores = softmax(Q @ K^T / √d)
     - Output = (Attention @ V) @ Wo
     - Residual: x = x + output

  2. Feedforward (SwiGLU):
     - RMSNorm(x)
     - gate = x @ W1
     - up = x @ W3
     - hidden = SiLU(gate) * up
     - output = hidden @ W2
     - Residual: x = x + output

Final: RMSNorm(x) → logits
```

### Key Components

**RMSNorm** (Root Mean Square Layer Normalization):
```
RMSNorm(x) = (x / RMS(x)) * weight
where RMS(x) = √(mean(x²) + ε)
```

**Multi-Head Attention**:
- Split queries, keys, values into N_HEADS heads
- Compute attention per head
- Concatenate and project

**SwiGLU Activation** (Swish-Gated Linear Unit):
```
SwiGLU(x) = SiLU(x @ W1) * (x @ W3)
where SiLU(x) = x * sigmoid(x)
```

## Configuration

```c
#define DIM 64              // Model dimension
#define HIDDEN_DIM 256      // FFN hidden dimension
#define N_LAYERS 2          // Number of transformer layers
#define N_HEADS 4           // Number of attention heads
#define N_KV_HEADS 4        // Number of key/value heads (GQA)
#define VOCAB_SIZE 256      // Vocabulary size
#define SEQ_LEN 32          // Sequence length
```

## Model Parameters

**Weights**:
- Token embeddings: `[VOCAB_SIZE × DIM]`
- Per-layer attention: `Wq, Wk, Wv, Wo`
- Per-layer FFN: `W1, W2, W3`
- RMSNorm weights for attention and FFN

**Total parameters**: ~200K (in mini config)

## Command-Line Usage

```bash
./transformer
```

The benchmark runs a forward pass through the transformer for a fixed sequence.

## Use in AI/ML

Transformer architecture is the foundation of:
- **Language Models**: GPT-2, GPT-3, GPT-4
- **Llama Family**: Llama, Llama2, Code Llama
- **Open Models**: Mistral, Mixtral, Phi
- **Instruction Models**: Alpaca, Vicuna, ChatGPT

## CARTS Compatibility

- No global variables (weights passed as parameters)
- Clean parameter passing via structs
- OpenMP parallel for in key operations:
  - Matrix multiplications
  - RMSNorm computation
  - Attention score computation
- Self-contained implementation

## Key Features

- **Minimal dependencies**: Only stdlib, math.h, omp.h
- **Educational focus**: Clear, readable implementation
- **Llama2 architecture**: RMSNorm, SwiGLU, GQA support
- **Grouped Query Attention (GQA)**: Memory-efficient attention
- **In-place operations**: Memory-efficient computation

## CARTS Testing Focus

### Memory Access Patterns
- **Sequential**: Token embeddings lookup
- **Strided**: Multi-head attention splits
- **Reduction**: RMSNorm statistics computation
- **Broadcast**: Attention scores across sequence

### Dependencies
- **Layer-to-layer**: Output of layer L feeds into L+1
- **Within attention**: Scores depend on Q, K computation
- **Residual connections**: Output depends on input + transformation

### Parallelization Opportunities
- **Across heads**: Attention heads are independent
- **Across sequence**: Some operations vectorize over sequence
- **Matrix operations**: GEMM operations (largest compute)

## Performance Characteristics

- **Compute**: Dominated by matrix multiplications (O(n²d) for attention, O(nd²) for FFN)
- **Memory**: Layer-by-layer computation, KV cache for autoregressive generation
- **Bottleneck**: Attention O(n²) complexity for long sequences

## References

- **llama2.c**: https://github.com/karpathy/llama2.c
- **Llama2 Paper**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
- **Transformer**: "Attention is All You Need" (Vaswani et al., 2017)
- **RMSNorm**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- **SwiGLU**: "GLU Variants Improve Transformer" (Shazeer, 2020)

## Citation

### llama2.c
```
Karpathy, Andrej. "llama2.c: Inference Llama 2 in one file of pure C"
https://github.com/karpathy/llama2.c, 2023.
```

### Llama2
```
Touvron, Hugo, et al.
"Llama 2: Open foundation and fine-tuned chat models."
arXiv preprint arXiv:2307.09288 (2023).
```

### Transformer Architecture
```
Vaswani, Ashish, et al.
"Attention is all you need."
Advances in neural information processing systems 30 (2017).
```

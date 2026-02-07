################################################################################
# Llama2 Transformer benchmark
################################################################################

EXAMPLE_NAME := transformer
SRC := transformer.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): LLaMA-3B shape. matmul=2048 (32x64), attention=32 heads. SEQ=64
# EXTRALARGE (1920-worker): DIM=1920 (1x1920), FFN=5760 (3x1920). ~5.7x LARGE
SMALL_CFLAGS ?= -DDIM=64 -DHIDDEN_DIM=256
MEDIUM_CFLAGS ?= -DDIM=128 -DHIDDEN_DIM=512
LARGE_CFLAGS ?= -DDIM=2048 -DHIDDEN_DIM=5504 -DN_HEADS=32 -DN_KV_HEADS=8 -DVOCAB_SIZE=32000 -DSEQ_LEN=64 -DN_LAYERS=4
EXTRALARGE_CFLAGS ?= -DDIM=1920 -DHIDDEN_DIM=5760 -DN_HEADS=30 -DN_KV_HEADS=6 -DVOCAB_SIZE=32000 -DSEQ_LEN=256 -DN_LAYERS=6

include ../common/carts.mk

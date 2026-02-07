################################################################################
# Llama2 Transformer benchmark
################################################################################

EXAMPLE_NAME := transformer
SRC := transformer.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): LLaMA-7B shape. matmul=4096 (64x64), attention=64 heads. SEQ=128
# EXTRALARGE (1920-worker): matmul=3840 (2x1920), FFN=11520 (6x1920). ~6x LARGE
SMALL_CFLAGS ?= -DDIM=64 -DHIDDEN_DIM=256
MEDIUM_CFLAGS ?= -DDIM=128 -DHIDDEN_DIM=512
LARGE_CFLAGS ?= -DDIM=4096 -DHIDDEN_DIM=11008 -DN_HEADS=64 -DN_KV_HEADS=8 -DVOCAB_SIZE=32000 -DSEQ_LEN=128 -DN_LAYERS=4
EXTRALARGE_CFLAGS ?= -DDIM=3840 -DHIDDEN_DIM=11520 -DN_HEADS=60 -DN_KV_HEADS=12 -DVOCAB_SIZE=32000 -DSEQ_LEN=512 -DN_LAYERS=6

include ../common/carts.mk

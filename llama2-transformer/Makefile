################################################################################
# Llama2 Transformer benchmark
################################################################################

EXAMPLE_NAME := transformer
SRC := transformer.c
LDFLAGS := -lm

# Size configurations - can be overridden via command line
# small: dim=64, hidden=256, medium: dim=128, hidden=512
# large: dim=256, hidden=1024, extralarge: dim=512, hidden=2048
SMALL_CFLAGS ?= -DDIM=64 -DHIDDEN_DIM=256
MEDIUM_CFLAGS ?= -DDIM=128 -DHIDDEN_DIM=512
LARGE_CFLAGS ?= -DDIM=256 -DHIDDEN_DIM=1024
EXTRALARGE_CFLAGS ?= -DDIM=512 -DHIDDEN_DIM=2048

include ../common/carts.mk

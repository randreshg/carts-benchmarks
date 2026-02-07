EXAMPLE_NAME := activations
SRC := activations.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 384M=64x6291456. EXTRALARGE (1920-worker): ~1.88G=1920x1048576. 5x LARGE
SMALL_CFLAGS ?= -DSIZE=262144
MEDIUM_CFLAGS ?= -DSIZE=1048576
LARGE_CFLAGS ?= -DSIZE=402653184
EXTRALARGE_CFLAGS ?= -DSIZE=2013265920

include ../carts.mk

EXAMPLE_NAME := activations
SRC := activations.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 75M (1.2Mx64). EXTRALARGE (1920-worker): 300M (156Kx1920). 4.0x LARGE
SMALL_CFLAGS ?= -DSIZE=262144
MEDIUM_CFLAGS ?= -DSIZE=1048576
LARGE_CFLAGS ?= -DSIZE=75000000 -DNREPS=70
EXTRALARGE_CFLAGS ?= -DSIZE=300000000 -DNREPS=70

include ../carts.mk

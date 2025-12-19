EXAMPLE_NAME := activations
SRC := activations.c
LDFLAGS := -lm

# Size configurations - can be overridden via command line
# SMALL: 256K elements, MEDIUM: 1M elements, LARGE: 4M elements, EXTRALARGE: 16M elements
SMALL_CFLAGS ?= -DSIZE=262144
MEDIUM_CFLAGS ?= -DSIZE=1048576
LARGE_CFLAGS ?= -DSIZE=4194304
EXTRALARGE_CFLAGS ?= -DSIZE=16777216

include ../carts.mk

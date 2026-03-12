EXAMPLE_NAME := jacobi2d
SRC := jacobi2d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for: i from 1 to N-2, so N-2 iterations
# LARGE (64-core): N=8192 -> 8190 iter (128x64). EXTRALARGE (1920-worker): N=16384. 4.0x LARGE
SMALL_CFLAGS ?= -DN=256 -DTSTEPS=50
MEDIUM_CFLAGS ?= -DN=1024 -DTSTEPS=100
LARGE_CFLAGS ?= -DN=8192 -DTSTEPS=1000
EXTRALARGE_CFLAGS ?= -DN=16384 -DTSTEPS=1000

include ../carts.mk

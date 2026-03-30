EXAMPLE_NAME := activations
SRC := activations.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line.
# Keep LARGE suitable for full 1..64 thread sweeps: enough work to expose
# scaling, but not so much that the 1-thread baseline times out or dominates
# the whole suite. The large size also matches the benchmark README.
SMALL_CFLAGS ?= -DSIZE=262144
MEDIUM_CFLAGS ?= -DSIZE=1048576
LARGE_CFLAGS ?= -DSIZE=16777216 -DNREPS=50
EXTRALARGE_CFLAGS ?= -DSIZE=300000000 -DNREPS=70

include ../carts.mk

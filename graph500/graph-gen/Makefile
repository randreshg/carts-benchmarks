EXAMPLE_NAME := graph_gen
SRC := graph_gen.c
LDFLAGS := -lm -ldl

# Size configurations for Graph500-inspired generation
# SMALL: 2^20 vertices (~17M edges, ~130 MB)
# MEDIUM: 2^24 vertices (~268M edges, ~2 GB)
# LARGE: 2^26 vertices (~1B edges, ~8 GB)
# EXTRALARGE: 2^28 vertices (~4B edges, ~32 GB) - exceeds single-node memory
SMALL_CFLAGS ?= -DSCALE=20 -DEDGE_FACTOR=16
MEDIUM_CFLAGS ?= -DSCALE=24 -DEDGE_FACTOR=16
LARGE_CFLAGS ?= -DSCALE=26 -DEDGE_FACTOR=16
EXTRALARGE_CFLAGS ?= -DSCALE=28 -DEDGE_FACTOR=16

include ../carts.mk

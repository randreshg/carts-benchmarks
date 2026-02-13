EXAMPLE_NAME := rhs4sg_base
SRC := rhs4sg_base.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for: k from 4 to NZ-4, so NZ-8 iterations
# LARGE (64-core): NZ=520 -> 512 k-iter (8x64). EXTRALARGE (1920-worker): NZ=1928 -> 1920 k-iter. 5.0x LARGE
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=40 -DNY=40 -DNZ=40
LARGE_CFLAGS ?= -DNX=512 -DNY=512 -DNZ=512
EXTRALARGE_CFLAGS ?= -DNX=512 -DNY=512 -DNZ=1920

include ../carts.mk

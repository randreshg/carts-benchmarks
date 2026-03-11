EXAMPLE_NAME := stress_update
SRC := stress_update.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for: k from 2 to NZ-2, so NZ-4 iterations
# LARGE (64-core): NZ=384 -> 380 k-iter. EXTRALARGE (1920-worker): NZ=1920. 5.0x LARGE
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=64 -DNY=64 -DNZ=64
LARGE_CFLAGS ?= -DNX=288 -DNY=288 -DNZ=384 -DNREPS=10
EXTRALARGE_CFLAGS ?= -DNX=288 -DNY=288 -DNZ=1920 -DNREPS=10

include ../carts.mk

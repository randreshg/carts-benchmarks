EXAMPLE_NAME := velocity_update
SRC := velocity_update.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for: k from 1 to NZ-2, so NZ-2 iterations
# LARGE (64-core): NZ=386 -> 384 k-iter (6x64). EXTRALARGE (1920-worker): NZ=1922 -> 1920 k-iter. 5.0x LARGE
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=64 -DNY=64 -DNZ=64
LARGE_CFLAGS ?= -DNX=386 -DNY=386 -DNZ=386
EXTRALARGE_CFLAGS ?= -DNX=386 -DNY=386 -DNZ=1922

include ../carts.mk

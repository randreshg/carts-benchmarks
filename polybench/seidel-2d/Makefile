EXAMPLE_NAME := seidel-2d
SRC := seidel-2d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Hot loop: N-2 iterations. LARGE (64-core): 9600 (150x64). EXTRALARGE (1920-worker): 19200 (10x1920). 4.0x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DTSTEPS=320 -DN=9600
EXTRALARGE_CFLAGS ?= -DTSTEPS=320 -DN=19200

include ../carts.mk

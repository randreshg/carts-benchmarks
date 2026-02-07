EXAMPLE_NAME := seidel-2d
SRC := seidel-2d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Hot loop: N-2 iterations. LARGE (64-core): 10048 (157x64). EXTRALARGE (1920-worker): 23040 (12x1920). 5.26x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DTSTEPS=250 -DN=10050
EXTRALARGE_CFLAGS ?= -DTSTEPS=250 -DN=23042

include ../carts.mk

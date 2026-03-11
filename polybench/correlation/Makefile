EXAMPLE_NAME := correlation
SRC := correlation.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 4800 (75x64). EXTRALARGE (1920-worker): 7680 (4x1920). 4.1x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DM=4800 -DN=4800
EXTRALARGE_CFLAGS ?= -DM=7680 -DN=7680

include ../carts.mk

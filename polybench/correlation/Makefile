EXAMPLE_NAME := correlation
SRC := correlation.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 3200 (50x64). EXTRALARGE (1920-worker): 5760 (3x1920). 5.83x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DM=3200 -DN=3200
EXTRALARGE_CFLAGS ?= -DM=5760 -DN=5760

include ../carts.mk

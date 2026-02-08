EXAMPLE_NAME := correlation
SRC := correlation.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 3200 (50x64). EXTRALARGE (1920-worker): 5760 (3x1920). 5.83x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DM=4800 -DN=4800
EXTRALARGE_CFLAGS ?= -DM=9600 -DN=9600

include ../carts.mk

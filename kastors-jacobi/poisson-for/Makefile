EXAMPLE_NAME := poisson-for
SRC := poisson-for.c
CFLAGS += -I../include
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for. T ~ SIZE^2. Same sizing as jacobi-for.
# LARGE (64-core): 15360 (240x64). EXTRALARGE (1920-worker): 30720 (16x1920). 4.0x LARGE
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=10112 -DNREPS=10
EXTRALARGE_CFLAGS ?= -DSIZE=21120 -DNREPS=10

include ../carts.mk

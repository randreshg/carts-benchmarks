EXAMPLE_NAME := poisson-for
SRC := poisson-for.c
CFLAGS += -I../include
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for. T ~ SIZE^2. Same sizing as jacobi-for.
# LARGE (64-core): 20480 (320x64). EXTRALARGE (1920-worker): 32640 (17x1920). 2.54x LARGE
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=20480
EXTRALARGE_CFLAGS ?= -DSIZE=32640

include ../carts.mk

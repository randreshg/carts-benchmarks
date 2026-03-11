EXAMPLE_NAME := jacobi-for
SRC := jacobi-for.c
CFLAGS += -I../include
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 19200 (300x64). EXTRALARGE (1920-worker): 38400 (20x1920). 4.0x LARGE
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=10240 -DNREPS=10
EXTRALARGE_CFLAGS ?= -DSIZE=21120 -DNREPS=10

include ../carts.mk

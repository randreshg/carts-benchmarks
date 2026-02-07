EXAMPLE_NAME := jacobi-for
SRC := jacobi-for.c
CFLAGS += -I../include
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 20480 (320x64). EXTRALARGE (1920-worker): 46080 (24x1920). 5.07x LARGE
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=20480
EXTRALARGE_CFLAGS ?= -DSIZE=46080

include ../carts.mk

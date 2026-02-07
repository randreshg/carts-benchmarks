EXAMPLE_NAME := poisson-task
SRC := poisson-task.c
CFLAGS += -I../include
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Task parallelism + parallel for (line 136). T ~ SIZE^2.
# LARGE (64-core): 8192 (128x64). EXTRALARGE (1920-worker): 19200 (10x1920). 5.49x LARGE
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=8192
EXTRALARGE_CFLAGS ?= -DSIZE=19200

include ../carts.mk

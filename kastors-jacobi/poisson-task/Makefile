EXAMPLE_NAME := poisson-task
SRC := poisson-task.c
CFLAGS += -I../include
LDFLAGS := -lm

# Size configurations - can be overridden via command line
# SMALL: 256x256, MEDIUM: 1024x1024, LARGE: 2048x2048, EXTRALARGE: 4096x4096
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=2048
EXTRALARGE_CFLAGS ?= -DSIZE=12288

include ../carts.mk

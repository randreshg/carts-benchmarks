EXAMPLE_NAME := poisson-task
SRC := poisson-task.c
CFLAGS += -I../include
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Task parallelism + parallel for (line 136). T ~ SIZE^2.
# LARGE (64-core): 10240 (160x64). EXTRALARGE (1920-worker): 23040 (12x1920). 5.06x LARGE
SMALL_CFLAGS ?= -DSIZE=256
MEDIUM_CFLAGS ?= -DSIZE=1024
LARGE_CFLAGS ?= -DSIZE=10240
EXTRALARGE_CFLAGS ?= -DSIZE=23040

include ../carts.mk

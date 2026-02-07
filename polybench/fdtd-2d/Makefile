EXAMPLE_NAME := fdtd-2d
SRC := fdtd-2d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): TMAX=100, N=4096 (64x64). EXTRALARGE (1920-worker): TMAX=500, N=3840 (2x1920). 4.39x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DTMAX=100 -DNX=4096 -DNY=4096
EXTRALARGE_CFLAGS ?= -DTMAX=500 -DNX=3840 -DNY=3840

include ../carts.mk

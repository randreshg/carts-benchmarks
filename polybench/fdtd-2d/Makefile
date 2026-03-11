EXAMPLE_NAME := fdtd-2d
SRC := fdtd-2d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): TMAX=100, NX=NY=3840 (60x64). EXTRALARGE (1920-worker): NX=NY=7680 (4x1920). 4.0x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DTMAX=100 -DNX=3840 -DNY=3840
EXTRALARGE_CFLAGS ?= -DTMAX=100 -DNX=7680 -DNY=7680

include ../carts.mk

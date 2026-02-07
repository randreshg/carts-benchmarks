EXAMPLE_NAME := convolution-2d
SRC := convolution-2d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 38400 (600x64). EXTRALARGE (1920-worker): 76800 (40x1920). 4x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DNI=38400 -DNJ=38400
EXTRALARGE_CFLAGS ?= -DNI=76800 -DNJ=76800

include ../carts.mk

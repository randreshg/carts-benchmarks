EXAMPLE_NAME := convolution-3d
SRC := convolution-3d.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 1024 (16x64). EXTRALARGE (1920-worker): asymmetric NI=1920. 4.22x LARGE
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DNI=512 -DNJ=512 -DNK=512 -DNREPS=180
EXTRALARGE_CFLAGS ?= -DNI=1920 -DNJ=512 -DNK=512 -DNREPS=180

include ../carts.mk

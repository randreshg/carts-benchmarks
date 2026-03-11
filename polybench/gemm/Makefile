EXAMPLE_NAME := gemm
SRC := gemm.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 4800 (75x64). EXTRALARGE (1920-worker): 7680 (4x1920). 4.1x LARGE
SMALL_CFLAGS ?= -DNI=128 -DNJ=128 -DNK=128
MEDIUM_CFLAGS ?= -DNI=1024 -DNJ=1024 -DNK=1024
LARGE_CFLAGS ?= -DNI=4800 -DNJ=4800 -DNK=4800
EXTRALARGE_CFLAGS ?= -DNI=7680 -DNJ=7680 -DNK=7680

include ../carts.mk

EXAMPLE_NAME := vel4sg_base
SRC := vel4sg_base.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for: k from 2 to NZ-2, so NZ-4 iterations
# LARGE (64-core): NZ=516 -> 512 k-iter (8x64). EXTRALARGE (1920-worker): NZ=1924 -> 1920 k-iter. 5.01x LARGE
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=40 -DNY=40 -DNZ=40
LARGE_CFLAGS ?= -DNX=516 -DNY=516 -DNZ=516
EXTRALARGE_CFLAGS ?= -DNX=596 -DNY=596 -DNZ=1924

include ../carts.mk

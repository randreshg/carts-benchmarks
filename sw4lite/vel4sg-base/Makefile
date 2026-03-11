EXAMPLE_NAME := vel4sg_base
SRC := vel4sg_base.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Parallel for: k from 2 to NZ-2, so NZ-4 iterations
# LARGE (64-core): NZ=448 -> 444 k-iter (6.9x64). EXTRALARGE (1920-worker): NZ=1920. 4.29x LARGE
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=40 -DNY=40 -DNZ=40
LARGE_CFLAGS ?= -DNX=320 -DNY=320 -DNZ=448 -DNREPS=10
EXTRALARGE_CFLAGS ?= -DNX=320 -DNY=320 -DNZ=1920 -DNREPS=10

include ../carts.mk

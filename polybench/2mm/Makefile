EXAMPLE_NAME := 2mm
SRC := 2mm.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE: NI=NJ=NK=NL=3600. EXTRALARGE: NI=NJ=NK=NL=5760
SMALL_CFLAGS ?= -DSMALL_DATASET
MEDIUM_CFLAGS ?= -DSTANDARD_DATASET
LARGE_CFLAGS ?= -DNI=3600 -DNJ=3600 -DNK=3600 -DNL=3600
EXTRALARGE_CFLAGS ?= -DNI=5760 -DNJ=5760 -DNK=5760 -DNL=5760
# MEGALARGE: 2x extralarge linear dim for 1n->2n communication amortization.
MEGALARGE_CFLAGS ?= -DNI=11520 -DNJ=11520 -DNK=11520 -DNL=11520

include ../carts.mk

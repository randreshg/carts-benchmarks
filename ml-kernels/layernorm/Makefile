EXAMPLE_NAME := layernorm
SRC := layernorm.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Total elements = BATCH * HIDDEN. Hot loop: parallel for over BATCH
# LARGE (64-core): BATCH=65536 (1024x64). EXTRALARGE (1920-worker): BATCH=245760 (128x1920). 3.75x LARGE
SMALL_CFLAGS ?= -DBATCH=64 -DHIDDEN=2048
MEDIUM_CFLAGS ?= -DBATCH=256 -DHIDDEN=4096
LARGE_CFLAGS ?= -DBATCH=65536 -DHIDDEN=8192 -DNREPS=130
EXTRALARGE_CFLAGS ?= -DBATCH=245760 -DHIDDEN=8192 -DNREPS=130

include ../carts.mk

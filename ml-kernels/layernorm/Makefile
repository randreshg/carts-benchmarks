EXAMPLE_NAME := layernorm
SRC := layernorm.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# Total elements = BATCH * HIDDEN. Hot loop: parallel for over BATCH
# LARGE (64-core): BATCH=262144 (4096x64), HIDDEN=8192 (LLaMA-70B class). ~2.15B work
# EXTRALARGE (1920-worker): BATCH=983040 (512x1920). 3.75x LARGE
SMALL_CFLAGS ?= -DBATCH=64 -DHIDDEN=2048
MEDIUM_CFLAGS ?= -DBATCH=256 -DHIDDEN=4096
LARGE_CFLAGS ?= -DBATCH=262144 -DHIDDEN=8192
EXTRALARGE_CFLAGS ?= -DBATCH=983040 -DHIDDEN=8192

include ../carts.mk

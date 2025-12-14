################################################################################
# Llama2 Transformer benchmark
################################################################################

EXAMPLE_NAME := transformer
SRC := transformer.c

# Size configurations
# small: dim=64, hidden=256 (default)
# medium: dim=128, hidden=512
# large: dim=256, hidden=1024
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with small config (dim=64, hidden=256)"
	$(MAKE) all openmp CFLAGS="-DDIM=64 -DHIDDEN_DIM=256" LDFLAGS="$(LDFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with medium config (dim=128, hidden=512)"
	$(MAKE) all openmp CFLAGS="-DDIM=128 -DHIDDEN_DIM=512" LDFLAGS="$(LDFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with large config (dim=256, hidden=1024)"
	$(MAKE) all openmp CFLAGS="-DDIM=256 -DHIDDEN_DIM=1024" LDFLAGS="$(LDFLAGS)"

LDFLAGS := -lm
include ../common/carts.mk

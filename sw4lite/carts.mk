################################################################################
# SW4Lite defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

SW4LITE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(SW4LITE_ROOT)/..)

# Shared includes for SW4Lite suite
INCLUDES ?= -I$(SW4LITE_ROOT)/common

# Auto-detect arts.cfg at suite level
ifneq ($(wildcard $(SW4LITE_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(SW4LITE_ROOT)/arts.cfg
endif

include $(BENCHMARKS_ROOT)/common/carts.mk

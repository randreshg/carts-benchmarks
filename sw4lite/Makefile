################################################################################
# SW4Lite Suite Makefile - Builds all SW4Lite benchmark variants
################################################################################

EXAMPLES := rhs4sg-base vel4sg-base
ifneq ($(wildcard rhs4sg-revnw),)
EXAMPLES += rhs4sg-revnw
endif

.PHONY: all clean $(EXAMPLES)
.PHONY: arts-wrapper openmp-wrapper
.PHONY: small medium large extralarge
.PHONY: small-arts medium-arts large-arts extralarge-arts
.PHONY: small-openmp medium-openmp large-openmp extralarge-openmp

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "Building SW4Lite/$@..."
	@$(MAKE) -C $@ all

small medium large extralarge:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Building SW4Lite/$$dir ($@)..."; \
			$(MAKE) -C $$dir $@; \
		fi; \
	done

small-arts medium-arts large-arts extralarge-arts:
	@size=$(@:-arts=); \
	for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Building SW4Lite/$$dir ($$size-arts)..."; \
			$(MAKE) -C $$dir $$size-arts; \
		fi; \
	done
	@$(MAKE) arts-wrapper

small-openmp medium-openmp large-openmp extralarge-openmp:
	@size=$(@:-openmp=); \
	for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Building SW4Lite/$$dir ($$size-openmp)..."; \
			$(MAKE) -C $$dir $$size-openmp; \
		fi; \
	done
	@$(MAKE) openmp-wrapper

arts-wrapper:
	@echo "Creating SW4Lite ARTS wrapper..."
	@rm -f sw4lite_arts
	@printf '#!/bin/bash\nset -e\n' > sw4lite_arts
	@for dir in $(EXAMPLES); do \
		exe=$$(ls "$$dir"/*_arts 2>/dev/null | head -n1); \
		if [ -n "$$exe" ]; then \
			printf '%s\n' "$$exe" >> sw4lite_arts; \
		fi; \
	done
	@chmod +x sw4lite_arts

openmp-wrapper:
	@echo "Creating SW4Lite OpenMP wrapper..."
	@mkdir -p build
	@rm -f build/sw4lite_omp
	@printf '#!/bin/bash\nset -e\n' > build/sw4lite_omp
	@for dir in $(EXAMPLES); do \
		exe=$$(ls "$$dir"/build/*_omp 2>/dev/null | head -n1); \
		if [ -n "$$exe" ]; then \
			printf '../%s\n' "$$exe" >> build/sw4lite_omp; \
		fi; \
	done
	@chmod +x build/sw4lite_omp

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Cleaning SW4Lite/$$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done

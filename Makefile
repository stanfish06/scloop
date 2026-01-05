.PHONY: build build-m4ri build-gf2toolkit sync clean

PROJECT_ROOT := $(shell pwd)

M4RI_SRC := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2/m4ri
M4RI_PREFIX := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2
GF2TOOLKIT_SRC := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2/GF2toolkit
GF2TOOLKIT_PREFIX := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2
DM_PREFIX := $(PROJECT_ROOT)/src/scloop/utils/distance_metrics

build-m4ri:
	cd $(M4RI_SRC) && \
		autoreconf -i && \
		./configure --prefix=$(M4RI_PREFIX) --libdir=$(M4RI_PREFIX) \
					--enable-static --disable-shared --enable-openmp \
					CFLAGS="-fPIC -O3" && \
		$(MAKE) && \
		$(MAKE) install

build-gf2toolkit: build-m4ri
	cd $(GF2TOOLKIT_SRC) && \
		$(MAKE) clean || true && \
		$(MAKE) libGF2toolkit.a INC="-Isrcs -I$(M4RI_PREFIX)/include" AR="ar rcs"
	cp $(GF2TOOLKIT_SRC)/libGF2toolkit.a $(GF2TOOLKIT_PREFIX)/

build: build-m4ri build-gf2toolkit
	CFLAGS="-I$(M4RI_PREFIX)/include" \
	LDFLAGS="-L$(M4RI_PREFIX) -Wl,-rpath,$(M4RI_PREFIX)" \
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX) \
		uv build

fresh-sync: build-m4ri build-gf2toolkit
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX)/discrete-frechet-distance:$(GF2TOOLKIT_SRC)/srcs uv sync

sync: clean build-m4ri build-gf2toolkit
	python setup.py build_ext --inplace
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX)/discrete-frechet-distance:$(GF2TOOLKIT_SRC)/srcs uv sync

full-sync: build-m4ri build-gf2toolkit
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf ./src/scloop.egg-info
	rm -f uv.lock
	find ./src -name '*.so'  -type f -delete
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX)/discrete-frechet-distance:$(GF2TOOLKIT_SRC)/srcs uv sync

clean:
	rm -rf dist/ *.egg-info
	cd $(M4RI_SRC) && $(MAKE) clean || true
	cd $(GF2TOOLKIT_SRC) && $(MAKE) clean || true
	rm -rf $(M4RI_PREFIX)/include $(M4RI_PREFIX)/lib*.a $(M4RI_PREFIX)/lib*.so* $(M4RI_PREFIX)/lib*.la $(M4RI_PREFIX)/pkgconfig
	rm -f $(GF2TOOLKIT_PREFIX)/libGF2toolkit.a $(GF2TOOLKIT_PREFIX)/*cpython*.so*
	rm -f $(DM_PREFIX)/*.so

rebuild: clean build

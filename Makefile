.PHONY: build sync clean

PROJECT_ROOT := $(shell pwd)

build:
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data/ripser_lib uv build

sync:
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data/ripser_lib uv sync

clean:
	rm -rf dist/ *.egg-info

rebuild: clean build

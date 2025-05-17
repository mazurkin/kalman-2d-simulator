# makefile
#
# https://swcarpentry.github.io/make-novice/reference.html
# https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
# https://www.gnu.org/software/make/manual/make.html
# https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

SHELL := /bin/bash
ROOT  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

CONDA_ENV_NAME = kalman-2d-simulator

export PYTHONDONTWRITEBYTECODE    = 1
export PYTHONUNBUFFERED           = 1
export PYGAME_HIDE_SUPPORT_PROMPT = 1

# -----------------------------------------------------------------------------
# notebook
# -----------------------------------------------------------------------------

.DEFAULT_GOAL = run

.PHONY: run
run:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" --cwd "$(ROOT)/src" \
		python simulator.py

# -----------------------------------------------------------------------------
# conda environment
# -----------------------------------------------------------------------------

.PHONY: env-init
env-init:
	@conda create --yes --copy --name "$(CONDA_ENV_NAME)" \
		python=3.10.16 \
		conda-forge::poetry=1.8.5 \
		conda-forge::libstdcxx-ng=14.2.0

.PHONY: env-create
env-create:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry install --no-root --no-directory

.PHONY: env-update
env-update:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry update

.PHONY: env-list
env-list:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry show

.PHONY: env-remove
env-remove:
	@conda env remove --yes --name "$(CONDA_ENV_NAME)"

.PHONY: env-shell
env-shell:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		bash

.PHONY: env-info
env-info:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		conda info

# -----------------------------------------------------------------------------
# linters
# -----------------------------------------------------------------------------

.PHONY: lint-flake8
lint-flake8:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		flake8 src

.PHONY: lint
lint: lint-flake8

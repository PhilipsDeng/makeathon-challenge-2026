# make sure this is executed with bash
SHELL := /bin/bash


YELLOW := "\e[1;33m"
NC := "\e[0m"

PYTHON ?= $(shell command -v python3.10 || command -v python3 || command -v python)

# Logger function
INFO := @bash -c '\
  printf $(YELLOW); \
  echo "=> $$1"; \
  printf $(NC)' SOME_VALUE

.venv:  # creates .venv folder if does not exist
	@if [ -z "$(PYTHON)" ]; then \
		echo "No Python interpreter found. Please install Python 3 and rerun 'make install'."; \
		exit 1; \
	fi
	$(PYTHON) -m venv .venv


.venv/bin/uv: .venv # installs latest pip
	.venv/bin/pip install -U uv

install: .venv/bin/uv
	# before running install cmake
	.venv/bin/python -m uv pip install -r requirements.txt
	# after installing source .venv/bin/activate in your shell

DOWNLOAD_ANON ?=

download_data_from_s3:
	.venv/bin/python -m download_data $(if $(DOWNLOAD_ANON),--anon)
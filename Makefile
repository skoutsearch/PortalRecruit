SHELL := /bin/bash

VENV ?= $(HOME)/.venvs/PortalRecruit
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help venv install run dashboard test clean

help:
	@echo "Targets:"
	@echo "  venv       - create venv at $(VENV) using python3.13 (skips if exists)"
	@echo "  install    - install python deps into venv"
	@echo "  run        - run run_portalrecruit.py"
	@echo "  dashboard  - run streamlit dashboard"
	@echo "  test       - run pytest (if present)"
	@echo "  clean      - remove local python caches"

venv:
	@if [[ -d "$(VENV)" ]]; then \
		echo "Venv exists at $(VENV) (set FORCE_RECREATE=1 to recreate)"; \
		if [[ "${FORCE_RECREATE:-0}" == "1" ]]; then rm -rf "$(VENV)"; else exit 0; fi; \
	fi
	python3.13 -m venv "$(VENV)"
	"$(PIP)" install -U pip

install: venv
	"$(PIP)" install -r requirements.txt

run:
	"$(PY)" run_portalrecruit.py

dashboard:
	"$(VENV)/bin/streamlit" run src/dashboard/Home.py

test:
	"$(VENV)/bin/pytest" -q || true

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete

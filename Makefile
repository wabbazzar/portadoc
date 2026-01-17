# Portadoc Makefile
# Run `make` or `make help` to see available commands

.PHONY: help install check extract extract-smart eval eval-smart serve clean

# Default target
.DEFAULT_GOAL := help

# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PORTADOC := $(VENV)/bin/portadoc
PDF ?= data/input/peter_lou_50dpi.pdf
GROUND_TRUTH ?= data/input/peter_lou_words_slim.csv
DPI ?= 300
OUTPUT ?=

##@ General

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install dependencies and setup venv
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
	fi
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
	$(VENV)/bin/pip install opencv-contrib-python
	@echo "Done! Run 'source $(VENV)/bin/activate' to activate the venv"

install-ocr: ## Install optional OCR dependencies (PaddleOCR, docTR, etc.)
	@if [ -f "tmp/install_ocr_deps.sh" ]; then \
		bash tmp/install_ocr_deps.sh; \
	else \
		echo "Installing OCR dependencies..."; \
		$(VENV)/bin/pip install paddleocr "python-doctr[torch]"; \
	fi

check: ## Check OCR engine availability
	$(PORTADOC) check

##@ Extraction

extract: ## Extract words from PDF (use PDF=path/to/file.pdf)
	$(PORTADOC) extract $(PDF) $(if $(OUTPUT),-o $(OUTPUT),)

extract-smart: ## Extract with smart harmonization (Tesseract-only, best F1)
	$(PORTADOC) extract --smart --no-easyocr --preprocess none --psm 6 $(PDF) $(if $(OUTPUT),-o $(OUTPUT),)

extract-all: ## Extract with ALL 4 engines (best text accuracy)
	$(PORTADOC) extract --smart --use-paddleocr --use-doctr --preprocess none --psm 6 $(PDF) $(if $(OUTPUT),-o $(OUTPUT),)

extract-clean: ## Extract from clean PDFs (high quality source)
	$(PORTADOC) extract --smart --no-easyocr --preprocess none --psm 6 $(PDF) $(if $(OUTPUT),-o $(OUTPUT),)

extract-verbose: ## Extract with progress bar
	$(PORTADOC) extract --smart --no-easyocr --preprocess none --psm 6 --progress $(PDF) $(if $(OUTPUT),-o $(OUTPUT),)

##@ Evaluation

eval: ## Evaluate extraction against ground truth
	$(PORTADOC) eval $(PDF) $(GROUND_TRUTH)

eval-smart: ## Evaluate with Tesseract-only (best F1)
	$(PORTADOC) eval --smart --no-easyocr --preprocess none --psm 6 $(PDF) $(GROUND_TRUTH)

eval-all: ## Evaluate with ALL 4 engines (best text accuracy)
	$(PORTADOC) eval --smart --use-paddleocr --use-doctr --preprocess none --psm 6 $(PDF) $(GROUND_TRUTH)

eval-verbose: ## Evaluate with detailed match info
	$(PORTADOC) eval --smart --no-easyocr --preprocess none --psm 6 --verbose $(PDF) $(GROUND_TRUTH)

benchmark: ## Run benchmark comparing Tess-only vs all engines
	@echo "=== Tesseract Only ===" && $(PORTADOC) eval --smart --no-easyocr --preprocess none --psm 6 $(PDF) $(GROUND_TRUTH) 2>&1 | grep -E "F1 Score|Text Match"
	@echo "=== All 4 Engines ===" && $(PORTADOC) eval --smart --use-paddleocr --use-doctr --preprocess none --psm 6 $(PDF) $(GROUND_TRUTH) 2>&1 | grep -E "F1 Score|Text Match"

##@ Server

serve: ## Start the web UI server (http://localhost:8000)
	$(PORTADOC) serve

serve-dev: ## Start server with auto-reload for development
	$(PORTADOC) serve --reload

serve-web: ## Alias for serve - launches web visualization UI
	@echo "Opening http://localhost:8000 in browser..."
	$(PORTADOC) serve

##@ Development

test: ## Run tests
	$(VENV)/bin/pytest tests/ -v

lint: ## Run linting
	$(VENV)/bin/ruff check src/

format: ## Format code
	$(VENV)/bin/ruff format src/

clean: ## Clean up temporary files and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

##@ Quick Commands

peter-lou: ## Extract from peter_lou_50dpi.pdf (degraded test file)
	$(PORTADOC) extract --smart --no-easyocr --preprocess none --psm 6 data/input/peter_lou_50dpi.pdf

peter-lou-eval: ## Evaluate peter_lou_50dpi.pdf against ground truth
	$(PORTADOC) eval --smart --no-easyocr --preprocess none --psm 6 data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv

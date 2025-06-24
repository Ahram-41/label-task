# Makefile for label-task

.PHONY: help install format lint clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies with uv"
	@echo "  format        - Format code with black and isort"
	@echo "  lint          - Check code formatting and style"
	@echo "  clean         - Clean up temporary files"

# Install dependencies with uv
install:
	uv sync --extra dev

# Format code
format:
	black ./src/ *.py
	isort ./src/ *.py

# Check formatting and linting
lint:
	black --check src/ *.py
	isort --check-only src/ *.py
	flake8 src/
	mypy src/

# Clean up
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +
	find . -name ".coverage" -delete
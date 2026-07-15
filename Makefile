.PHONY: proto lint test clean install

install:
	pip install -e ".[dev]"

lint:
	ruff check a2a/ tests/
	ruff format --check a2a/ tests/
	mypy a2a/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=a2a --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .ruff_cache/

proto:
	@echo "Generating protobuf code..."
	@mkdir -p a2a/protocol/generated
	python -m grpc_tools.protoc \
		-I a2a/protocol \
		--python_out=a2a/protocol/generated \
		--grpc_python_out=a2a/protocol/generated \
		a2a/protocol/a2a.proto 2>/dev/null || echo "(proto not yet defined — Sprint 1)"

pre-commit:
	pre-commit run --all-files

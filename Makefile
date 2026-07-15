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
	@touch a2a/protocol/generated/__init__.py
	python -m grpc_tools.protoc \
		-I a2a/protocol \
		--python_out=a2a/protocol/generated \
		--grpc_python_out=a2a/protocol/generated \
		a2a/protocol/a2a.proto
	@sed -i 's/import a2a_pb2 as/from a2a.protocol.generated import a2a_pb2 as/' a2a/protocol/generated/a2a_pb2_grpc.py
	@echo "Proto codegen complete."

pre-commit:
	pre-commit run --all-files

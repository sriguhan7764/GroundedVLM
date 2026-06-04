.PHONY: install install-dev test lint format clean eval-coco benchmark

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,viz]"

test:
	pytest tests/ -v --cov=groundedvlm --cov-report=term-missing

lint:
	flake8 groundedvlm/ scripts/ tests/ --max-line-length=100
	mypy groundedvlm/ --ignore-missing-imports

format:
	black groundedvlm/ scripts/ tests/ --line-length=100
	isort groundedvlm/ scripts/ tests/

eval-coco:
	python scripts/eval_coco.py \
		--coco-root data/coco \
		--ann-file data/coco/annotations/instances_val2017.json \
		--detector dino \
		--output-dir outputs/coco_eval

benchmark:
	python scripts/benchmark_encoders.py \
		--encoders clip,align,siglip \
		--with-reranking \
		--output-dir outputs/retrieval_bench

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage

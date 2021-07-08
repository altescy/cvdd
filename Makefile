PWD              := $(shell pwd)
PYTHON           := poetry run python
PYTEST           := poetry run pytest
PYSEN            := poetry run pysen
MODULE           := colt

20NEWSGROUPS_DATASET := data/20newsgroups

.PHONY: datasets
datasets: $(20NEWSGROUPS_DATASET)

$(20NEWSGROUPS_DATASET):
	$(PWD)/scripts/20newsgroups/download.sh
	PYTHONPATH=$(PWD) $(PYTHON) scripts/20newsgroups/split_dataset.py data/raw/20_Newsgroups data/20newsgroups

.PHONY: lint
lint:
	$(PYSEN) run lint

.PHONY: format
format:
	$(PYSEN) run format

.PHONY: test
test:
	PYTHONPATH=$(PWD) $(PYTEST)

.PHONY: clean
clean: clean-pyc clean-build

.PHONY: clean-pyc
clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-build
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf pip-wheel-metadata/

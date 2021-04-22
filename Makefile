all: lint test dist 

.IGNORE: lint
.PHONY: lint
lint: 
	python setup.py lint

.PHONY: test
test:
	python setup.py pytest

.PHONY: dist
dist:
	python setup.py bdist_wheel

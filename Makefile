all: install lint docs test dist 

.PHONY: install
install:
	python setup.py install

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

.PHONY: docs
docs:
	python setup.py build_sphinx -c docs/
	sensible-browser build/sphinx/html/index.html

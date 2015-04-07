all: README.rst

# This contains all targets required before commit.
pre-commit: README.rst test

.PHONY: all test pre-commit

README.rst : docs/notebooks/README.ipynb
	ipython nbconvert --to=rst --output=$@ $<

test:
	python setup.py test

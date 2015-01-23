NOSETESTS = nosetests
NOSETESTS_FLAGS = --with-doctest --with-coverage --cover-package=pytimeode\
                  --cover-html --cover-html-dir=_build/coverage

all: README.rst

test:
	$(NOSETESTS) $(NOSETESTS_FLAGS)

# This contains all targets required before commit.
pre-commit: README.rst

.PHONY: all test pre-commit

README.rst : docs/notebooks/README.ipynb
	ipython nbconvert --to=rst --output=$@ $<

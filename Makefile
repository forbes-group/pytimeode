NOSETESTS = nosetests
NOSETESTS_FLAGS = --with-doctest --with-coverage --cover-package=pytimeode\
                  --cover-html --cover-html-dir=_build/coverage

all: README.rst

test:
	$(NOSETESTS) $(NOSETESTS_FLAGS)

.PHONY: all test

README.rst : docs/notebooks/README.ipynb
	ipython nbconvert --to=rst --output=$@ $<

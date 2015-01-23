
NOSETESTS = nosetests
NOSETESTS_FLAGS = --with-doctest --with-coverage --cover-package=pytimeode\
                  --cover-html --cover-html-dir=_build/coverage

test:
	$(NOSETESTS) $(NOSETESTS_FLAGS)

.PHONY: test

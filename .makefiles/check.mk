# Static analysis and checking (lint) 
#
# The check target allows you to configure what checks will be performed in your
# makefile and then use the make check target with flymake mode for example.
#
# Usage: Add a check target to your makefile with the appropriate dependencies.

TOP_DIR ?= $(realpath .)

ifeq (check,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  CHECK_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(CHECK_ARGS):;@:)
endif

ifeq ($(wildcard .pylintrc),)
else
  PYLINTRC ?= .pylintrc
endif

.PHONY: help help.check check
help: help.check
help.check:
	@echo
	@echo "          check"
	@echo "          -----"
	@echo " check <filename>  to run pylint etc. on filename."


.PHONY: check-pycheck check-pylint check-pyflakes check-pep8
check-pycheck:
	-python pycheck.py $(CHECK_ARGS) $(PYLINTRC)
check-pylint:
	-PYLINTRC=$(PYLINTRC) epylint $(CHECK_ARGS)
check-pyflakes:
	-pyflakes $(CHECK_ARGS)
check-pep8:
	-pep8 $(CHECK_ARGS)


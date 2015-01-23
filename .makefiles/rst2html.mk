# Documentation.  Automatic compilation of the README.rst files to aid editing.
# The watch target will continually run make, generating files as they change,
# optionally opening them in a browser.
#
# Requires:
# * RST2HTML = rst2html.py: Docutuils script for converting rst to html.
#
# Possible customization:
# * RST_HTML_OUTPUT: Set this to a new directory to collect all the html files
#   (organized in the same structure).  Default is to build in place.

# Include this first so that help is the first target
include $(makefiles_dir)/help.mk


RST2HTML ?= rst2html.py
README   ?= README.rst

TOP_DIR         ?= $(realpath .)
RST_HTML_INPUT  ?= $(TOP_DIR)
RST_HTML_OUTPUT ?= $(TOP_DIR)

uname = $(shell uname)

ifeq ($(uname), Darwin)
  AUTO_OPEN       ?= open $@ && sleep 1 && open $<
endif
WATCH ?= false

# Update the following to contain all files that should be built by watch.
all_readmes := $(shell find "$(RST_HTML_INPUT)" -type f -name "$(README)")
all_html = $(patsubst $(RST_HTML_INPUT)/%.rst, $(RST_HTML_OUTPUT)/%.html, $(all_readmes))

$(RST_HTML_OUTPUT)/%.html: $(RST_HTML_INPUT)/%.rst
	mkdir -p "$(dir $@)"
	$(RST2HTML) $< > $@
	$(WATCH) && ( $(AUTO_OPEN) ) || true

# This rule allows one to use a short form like make README.html.  It might
# guess incorrectly though.
%.html: $(RST_HTML_OUTPUT)/%.html
	@echo "Making $@ (assuming this is $<)"
	while sleep 1; do make -q $< || make WATCH=true $< ; done

.PHONY: watch rst-html
watch:
	while sleep 1; do make -q rst-html || make WATCH=true rst-html ; done

rst-html: $(all_html)

#################
# Help, cleaning, and debugging.
.PHONY: help.rst
help.rst:
	@echo
	@echo "          rst2html conversion: README.rst -> README.html"
	@echo "          -------------------"
	@echo " rst-html to make all README.html files"
	@echo " watch    to continuously make rst-html as README.rst changes"

.PHONY: clean.rst
clean.rst:
	rm -f $(all_html)

.PHONY: debug.rst
debug.rst:
	@echo RST_HTML_INPUT = "$(RST_HTML_INPUT)"
	@echo RST_HTML_OUTPUT = "$(RST_HTML_OUTPUT)"
	@echo all_readmes = "$(all_readmes)"
	@echo all_html = "$(all_html)"

.PHONY: help clean debug_makefile
help: help.rst
clean: clean.rst
debug_makefile: debug.rst

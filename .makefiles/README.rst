.. -*- rst -*- -*- restructuredtext -*-

.. This file should be written using the restructure text
.. conventions.  It will be displayed on the bitbucket source page and
.. serves as the documentation of the directory.

Makefiles: README branch
========================

A collection of Makefiles and related tools supporting the following features:

* Simple conversion of ``README.rst`` files to ``README.html`` files using
  docutils.

This branch only provides the README features that converts ``README.rst`` files
to ``README.html`` files for local viewing and the check feature for static
analysis of code.

Requirements
============
Requires the ``rst2html.py`` script which is provided by the docutils_ package.
(The script location can be customized by the ``RST2HTML`` variable.)

Usage
=====
A simple use (I have started using this for all my projects) is to include a
top-level ``Makefile`` of the form::

   makefiles_dir = .makefiles
   include $(makefiles_dir)/rst2html.mk

By default, this will convert ``README.rst`` files to ``README.html`` files in
the same directory (you might like to ignore these in your ``.hgignore`` file
for example).  You can customize this behaviour with the following variables:

   ``README (= README.rst)``:
      Name of input files.
   ``RST_HTML_INPUT``:
      Input directory.  This will be searched using ``find
      "$(RST_HTML_INPUT)" -type f -name "$(README)"``. 
   ``RST_HTML_OUTPUT``:
      Output directory (will have the same directory structure as the input
      directory.)
   ``RST2HTML (= rst2html.py)``:
       Name of ``rst2html.py`` command.

Static Checking
===============
If you want to include the static analysis checking, then make sure you have
the following in your ``Makefile``::

   makefiles_dir = .makefiles
   include $(makefiles_dir)/check.mk

   check: check-pylint check-pep8 ...

Add whatever static checkers you like.  Then you can run these (from flymake
mode in Emacs for example) with::

   $ make check tst.py

See the ``check.mk`` file for details about supported check targets and
how customize the behaviour by specifying rc files.

References
==========

* docutils_
* reStructuredText_
* `Quick reStructuredText`_

.. _docutils: http://docutils.sourceforge.net
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Quick reStructuredText: 
   http://docutils.sourceforge.net/docs/user/rst/quickref.html

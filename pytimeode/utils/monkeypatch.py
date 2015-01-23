"""This is the place to patch any bugs in other code.
"""

import logging


def _patch_zope_interface():
    """Provides a real "asStructuredText" replacement that produces
    reStructuredText so I can use it in README.rst etc.
    """
    import zope.interface.document
    from zope.interface.document import (_justify_and_indent, _trim_doc_string)

    def asStructuredText(I, munge=0):
        """ Output structured text format.  Note, this will whack any existing
        'structured' format of the text.  """

        r = ["``%s``" % (I.getName(),)]
        outp = r.append
        level = 1

        if I.getDoc():
            outp(_justify_and_indent(_trim_doc_string(I.getDoc()), level))

        bases = [base
                 for base in I.__bases__
                 if base is not zope.interface.Interface
                 ]
        if bases:
            outp(_justify_and_indent("This interface extends:", level, munge))
            level += 1
            for b in bases:
                item = "o ``%s``" % b.getName()
                outp(_justify_and_indent(_trim_doc_string(item), level, munge))
            level -= 1

        namesAndDescriptions = sorted(I.namesAndDescriptions())

        outp(_justify_and_indent("Attributes:", level, munge))
        level += 1
        for name, desc in namesAndDescriptions:
            if not hasattr(desc, 'getSignatureString'):   # ugh...
                item = "``%s`` -- %s" % (desc.getName(),
                                     desc.getDoc() or 'no documentation')
                outp(_justify_and_indent(_trim_doc_string(item), level, munge))
        level -= 1

        outp(_justify_and_indent("Methods:", level, munge))
        level += 1
        for name, desc in namesAndDescriptions:
            if hasattr(desc, 'getSignatureString'):   # ugh...
                item = "``%s%s`` -- %s" % (desc.getName(),
                                       desc.getSignatureString(),
                                       desc.getDoc() or 'no documentation')
                outp(_justify_and_indent(_trim_doc_string(item), level, munge))

        return "\n\n".join(r) + "\n\n"

    logging.info(
        "Patching zope.interface.document.asStructuredText to format code")
    zope.interface.document.asStructuredText = asStructuredText

_patch_zope_interface()

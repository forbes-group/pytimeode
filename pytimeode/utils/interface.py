"""Standing for zope.interface if it is not available."""

__all__ = ['Interface', 'Attribute', 'implements',
           'verifyObject', 'verifyClass']

import warnings

try:
    import zope.interface
    __all__ = zope.interface.__all__
    from zope.interface import (Interface, Attribute, implements)
    from zope.interface.verify import (verifyObject, verifyClass)
except ImportError:
    warnings.warn("Could not import ope.interface... using dummy stand-ins")

    Interface = object

    class Attribute(object):
        """Dummy"""
        def __init__(self, __name__, __doc__=''):
            pass

    def implements(*interfaces):
        """Dummy"""

    def verifyObject(iface, candidate):
        """Dummy"""

    def verifyClass(iface, candidate):
        """Dummy"""

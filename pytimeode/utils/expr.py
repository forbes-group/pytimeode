"""Interface to numexpr.

Allows numexpr to be applied to State objects.
"""
from __future__ import absolute_import

import collections

import numpy as np
import numexpr
import sympy

from mmfutils import interface

from .. import interfaces


class Expression(object):
    """Represent an expression to be computed efficiently.  Modeled on
    ``numexp.NumExpr()`` but with some slight variations:

    * Requires kwargs rather positional parameters as this is slightly safer in
      code.
    * Uses sympy to do some simplification and replacement of constants.
    * Deals with issue #81 by using the ``onejay`` symbol.

    Some limitations:

    * Do not use the following names for arguments or variables:
      `I`, `_onejay`, `out`

    Examples
    --------
    >>> np.random.seed(2)
    >>> a = np.random.rand(10)
    >>> expr = Expression('sin(a**3)', ['a'], dtype=a.dtype)
    >>> res = np.empty(a.shape, dtype=a.dtype)
    >>> res = expr(out=res, a=a)
    >>> np.allclose(res, np.sin(a**3))
    True
    """

    # Dictionary mapping numpy dtypes to numexpr kinds:
    dtype_to_type = dict(
        (np.dtype(_type), _type)
        for _type in numexpr.expressions.scalar_constant_types)

    def get_type(self, obj_or_type):
        try:
            dtype = getattr(obj_or_type, 'dtype', np.dtype(obj_or_type))
        except TypeError:
            dtype = np.dtype(type(obj_or_type))
        return self.dtype_to_type[dtype]

    def __init__(self, expr, args, state=None, dtype=float,
                 constants={}, simplify=True,
                 optimization='aggressive',
                 truediv='auto', ex_uses_vml=False,
                 kw={}):
        """First argument must be the expression, the remaining arguments
        define the types.  They can either be types or arrays (in which case
        the  ``dtype`` attribute will be used).

        Arguments
        ---------
        expr : str
           String representing the expression.
        args : [str]
           List of argument names with parameters appearing in the expression.
        state : INumexpr
           Instance of the state class implementing the IState interface.  If
           this is not provided, then the arguments will be assumed to be numpy
           arrays, otherwise the __call__ function will iterate.
        dtype : type
           If state is not provided, then this should be provided so the
           signatures can be generated.
        optimization, truediv :
           These are arguments for the numexpr compiler.  See the numexpr
           documentation or source code.
        kw : dict
           Additional kw arguments will be stored and passed to the call
           function.
        """
        if state is not None:
            interface.verifyObject(interfaces.INumexpr, state)
            dtype = state.dtype

        dtype = self.get_type(dtype)
        if isinstance(args, collections.Mapping):
            signature = [(_k, args.get(_k, dtype)) for _k in sorted(args)]
        else:
            signature = [(_k, dtype) for _k in sorted(args)]

        _onejay = sympy.S('_onejay')
        sexpr = sympy.S(expr).subs(constants).evalf()
        if simplify:
            sexpr = sympy.simplify(sexpr)

        F = sympy.Function
        sexpr = sexpr.subs([
            (sympy.I, _onejay),
            (sympy.Abs, F('abs')),
            (sympy.re, F('real')),
            (sympy.im, F('imag')),
            (sympy.acos, F('arccos')),
            (sympy.acosh, F('arccosh')),
            (sympy.asin, F('arcsin')),
            (sympy.asinh, F('arcsinh')),
            (sympy.atan, F('arctan')),
            (sympy.atan2, F('arctan2')),
            (sympy.atanh, F('arctanh')),
            (sympy.ln, F('log')),
        ])

        expr = str(sexpr)

        if '_onejay' in expr:
            signature.append(('_onejay', complex))

        types = collections.OrderedDict(signature)
        ast = numexpr.necompiler.expressionToAST(
            numexpr.necompiler.stringToExpression(expr, types, {}))
        variables = {}
        for a in ast.allOf('variable'):
            variables[a.value] = a
        variable_names = set(variables.keys())
        signature = [(_v, types[_v]) for _v in types if _v in variable_names]

        self._numexpr = numexpr.NumExpr(
            expr,
            signature=signature,
            optimization=optimization,
            truediv=truediv)

        self.signature = signature
        self.kw = dict(ex_uses_vml=False, **kw)
        self.expr = expr

    def __call__(self, out, **kw):
        """Default implementation valid only for arrays"""
        kw['_onejay'] = 1j
        kw['out'] = out
        args = [kw.pop(_k[0]) for _k in self.signature]
        kw.update(self.kw)
        return self._numexpr(*args, **kw)

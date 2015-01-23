"""Interface to numexpr.

Allows numexpr to be applied to State objects.
"""

import numpy as np
import numexpr
import sympy


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

    def __init__(self, expr, sig={}, state=None,
                 constants={}, simplify=True,
                 optimization='aggressive',
                 truediv='auto', ex_uses_vml=False,
                 kw={}):
        """First argument must be the expression, the remaining arguments
        define the types.  They can either be types or arrays (in which case
        the  ``dtype`` attribute will be used).

        Arguments
        ---------
        state : IState
           Instance of the state class implementing the IState interface.  If
           this is not provided, then the arguments will be assumed to be numpy
           arrays, otherwise the __call__ function will iterate
        optimization, truediv :
           These are arguments for the numexpr compiler.  See the numexpr
           documentation or source code.
        kw : dict
           Additional kw arguments will be stored and passed to the call
           function.
        """
        signature = [(_k, self.get_type(sig[_k])) for _k in sorted(sig)]

        _onejay = sympy.S('_onejay')
        sexpr = sympy.S(expr).subs(constants).evalf()
        if simplify:
            sexpr = sympy.simplify(sexpr).subs(sympy.I, _onejay)

        expr = str(sexpr)
        if '_onejay' in expr:
            signature.append(('_onejay', complex))

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


def test_expression():
    import numpy as np
    size = (100, 100)
    y0 = np.ones(size, dtype=complex) + 1j
    res = y0.copy()
    ans = (np.sin(y0)**2 + 1j)/5.0
    expr = Expression('(sin(y0)**2 + 1j)/5',
                      dict(y0=complex), ex_uses_vml=True)
    expr(y0=y0, out=res)
    assert np.allclose(res, ans)

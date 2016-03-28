import numpy as np

from ..utils import expr


class TestExpression(object):
    def test_expression_complex(self):
        size = (100, 100)
        y0 = np.ones(size, dtype=complex) + 1j
        res = y0.copy()
        ans = (np.sin(y0)**2 + 1j)/5.0
        e = expr.Expression('(sin(y0)**2 + 1j)/5',
                            dict(y0=complex), ex_uses_vml=True)
        e(y0=y0, out=res)
        assert np.allclose(res, ans)

    def test_expression_function_replacements(self):
        size = (100, 100)
        y0 = np.ones(size, dtype=complex) + 1j
        res = y0.copy()

        ans = (abs(y0) + np.arctan2(y0.real, 2*y0.imag) + 1j)/5.0
        e = expr.Expression('(Abs(y0) + atan2(re(y0), 2*im(y0)) + 1j)/5',
                            dict(y0=complex), ex_uses_vml=True)
        e(y0=y0, out=res)
        assert np.allclose(res, ans)

        for _f_sympy, _f_numpy in [
                ('Abs', abs),
                ('acos', np.arccos),
                ('acosh', np.arccosh),
                ('asin', np.arcsin),
                ('asinh', np.arcsinh),
                ('atan', np.arctan),
                ('atanh', np.arctanh),
                ('ln', np.log),
                ('re', np.real),
                ('im', np.imag)]:
            ans = (_f_numpy(y0)+1j)/5
            e = expr.Expression("({}(y0) + 1j)/5".format(_f_sympy),
                                dict(y0=complex), ex_uses_vml=True)
            e(y0=y0, out=res)
            assert np.allclose(res, ans)


class TestExpressionRegression(object):
    """Regression tests for various issues."""
    def test_expression_issue_8(self):
        size = (5, 5)
        y0 = np.ones(size)
        y1 = 2*y0
        res = y0.copy()
        ans = (np.sin(y0)**2 + 1.0)/5.0 + 0*y1
        e = expr.Expression('(sin(y0)**2 + 1.0)/5 + 0*y1',
                            ['y0', 'y1'], ex_uses_vml=True)
        e(y0=y0, y1=y1, out=res)
        assert np.allclose(res, ans)

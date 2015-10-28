import nose.tools as nt
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

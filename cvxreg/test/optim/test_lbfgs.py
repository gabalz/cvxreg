import unittest
import numpy as np

from cvxreg.optim.optim_task import OptimTask
from cvxreg.optim.lbfgs import LBFGS


class QuadFun(OptimTask):
    def fun(self, x):
        super().fun(x)
        return 0.5 * x.dot(x)

    def jac(self, x):
        super().jac(x)
        return x


class Himmelblau(OptimTask):
    def _cache(self, x):
        self.cache_key = x
        self.v1 = x[0] ** 2 + x[1] - 11.0
        self.v2 = x[0] + x[1] ** 2 - 7.0

    def fun(self, x):
        super().fun(x)
        self._cache(x)
        return self.v1 ** 2 + self.v2 ** 2

    def jac(self, x):
        super().jac(x)
        if not np.array_equiv(self.cache_key, x):
            self._cache(x)
        return np.array([
            4.0 * self.v1 * x[0] + 2.0 * self.v2,
            2.0 * self.v1 + 4.0 * self.v2 * x[1],
        ])


class TestLBFGSQuadratic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = LBFGS().minimize(QuadFun(), x0=np.ones(2))

    def test_status(self):
        self.assertEqual(self.res.status, 'Success')

    def test_solution(self):
        np.testing.assert_array_almost_equal(self.res.x, np.array([0., 0.]))

    def test_niter(self):
        self.assertEqual(self.res.niter, 2)

    def test_nrestarts(self):
        self.assertEqual(self.res.nrestarts, 0)


class TestLBFGSHimmelblau(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = LBFGS().minimize(Himmelblau(), x0=np.zeros(2))

    def test_status(self):
        self.assertEqual(self.res.status, 'Success')

    def test_solution(self):
        np.testing.assert_array_almost_equal(np.round(self.res.x, decimals=5), np.array([3., 2.]))

    def test_fval(self):
        self.assertAlmostEqual(np.round(self.res.fval, decimals=6), 0.0)

    def test_niter(self):
        self.assertEqual(self.res.niter, 11)

    def test_nrestarts(self):
        self.assertEqual(self.res.nrestarts, 1)

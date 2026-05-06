import unittest
import numpy as np
from scipy import sparse

from cvxreg.optim.quadprog import (
    qp_solve, convert_matrix_to_qp_solver_format,
    QP_BACKEND__OSQP, QP_BACKEND__CLARABEL,
)

_H1 = np.array([[3., 1.], [1., 1.]]) * 2.0
_g1 = np.array([1., 6.])
_A1 = np.array([[-2., -3.], [-1., 0.], [0., -1.]])
_b1 = np.array([-4., 0., 0.])

_H2 = sparse.csc_matrix([[4, 1], [1, 2]])
_g2 = np.array([1, 1])
_A2 = sparse.csc_matrix([[-1, -1], [-1, 0], [0, -1], [1, 1], [1, 0], [0, 1]])
_b2 = np.array([-1, 0, 0, 1, 0.7, 0.7])


class TestQuadProgOSQP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r1 = qp_solve(_H1, _g1, _A1, _b1, backend=QP_BACKEND__OSQP, verbose=False)
        cls.r2 = qp_solve(_H2, _g2, _A2, _b2, backend=QP_BACKEND__OSQP, verbose=False)
        x0 = np.array([0.2, 0.5])
        y0 = np.array([2.8, 0.0, 0.0, 0.0, 0.0, 0.3])
        cls.r3 = qp_solve(_H2, _g2, _A2, _b2, x0=x0, y0=y0,
                          backend=QP_BACKEND__OSQP, verbose=False)

    def test_problem1_solution(self):
        np.testing.assert_array_almost_equal(np.round(self.r1.primal_soln, decimals=6),
                                             np.array([0.5, 1.]))

    def test_problem1_dual(self):
        np.testing.assert_array_almost_equal(np.round(self.r1.dual_soln, decimals=6),
                                             np.array([3., 0., 0.]))

    def test_problem1_objective(self):
        x1 = self.r1.primal_soln
        self.assertAlmostEqual(
            np.round(0.5 * x1.dot(_H1.dot(x1)) + _g1.dot(x1), decimals=6), 9.25)

    def test_problem1_feasibility(self):
        self.assertAlmostEqual(
            np.round(np.max(_A1.dot(self.r1.primal_soln) - _b1), decimals=6), 0.0)

    def test_problem2_solution(self):
        np.testing.assert_array_almost_equal(np.round(self.r2.primal_soln, decimals=2),
                                             np.array([0.3, 0.7]))

    def test_problem2_dual(self):
        np.testing.assert_array_almost_equal(np.round(self.r2.dual_soln, decimals=1),
                                             np.array([2.9, 0., 0., 0., 0., 0.2]))

    def test_problem2_objective(self):
        x2 = self.r2.primal_soln
        self.assertAlmostEqual(
            np.round(0.5 * x2.dot(_H2.dot(x2)) + _g2.dot(x2), decimals=2), 1.88)

    def test_problem2_feasibility(self):
        self.assertAlmostEqual(
            np.round(np.max(_A2.dot(self.r2.primal_soln) - _b2), decimals=6), 0.0)

    def test_problem2_warm_start(self):
        np.testing.assert_array_almost_equal(np.round(self.r3.primal_soln, decimals=2),
                                             np.array([0.3, 0.7]))
        np.testing.assert_array_almost_equal(np.round(self.r3.dual_soln, decimals=1),
                                             np.array([2.9, 0., 0., 0., 0., 0.2]))


class TestQuadProgClarabel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H1c = convert_matrix_to_qp_solver_format(_H1, QP_BACKEND__CLARABEL)
        A1c = convert_matrix_to_qp_solver_format(_A1, QP_BACKEND__CLARABEL)
        cls.r1 = qp_solve(H1c, _g1, A1c, _b1, backend=QP_BACKEND__CLARABEL, verbose=False)
        cls.r2 = qp_solve(_H2, _g2, _A2, _b2, backend=QP_BACKEND__CLARABEL, verbose=False)
        x0 = np.array([0.2, 0.5])
        y0 = np.array([2.8, 0.0, 0.0, 0.0, 0.0, 0.3])
        cls.r3 = qp_solve(_H2, _g2, _A2, _b2, x0=x0, y0=y0,
                          backend=QP_BACKEND__CLARABEL, verbose=False)
        cls.H1c = H1c
        cls.A1c = A1c

    def test_problem1_solution(self):
        np.testing.assert_array_almost_equal(np.round(self.r1.primal_soln, decimals=6),
                                             np.array([0.5, 1.]))

    def test_problem1_dual(self):
        np.testing.assert_array_almost_equal(np.round(self.r1.dual_soln, decimals=6),
                                             np.array([3., 0., 0.]))

    def test_problem1_objective(self):
        x1 = self.r1.primal_soln
        self.assertAlmostEqual(
            np.round(0.5 * x1.dot(self.H1c.dot(x1)) + _g1.dot(x1), decimals=6), 9.25)

    def test_problem1_feasibility(self):
        self.assertAlmostEqual(
            np.round(np.max(self.A1c.dot(self.r1.primal_soln) - _b1), decimals=6), 0.0)

    def test_problem2_solution(self):
        np.testing.assert_array_almost_equal(np.round(self.r2.primal_soln, decimals=2),
                                             np.array([0.3, 0.7]))

    def test_problem2_dual(self):
        np.testing.assert_array_almost_equal(np.round(self.r2.dual_soln, decimals=1),
                                             np.array([4.2, 0., 0., 1.3, 0., 0.2]))

    def test_problem2_objective(self):
        x2 = self.r2.primal_soln
        self.assertAlmostEqual(
            np.round(0.5 * x2.dot(_H2.dot(x2)) + _g2.dot(x2), decimals=2), 1.88)

    def test_problem2_feasibility(self):
        self.assertLessEqual(np.max(_A2.dot(self.r2.primal_soln) - _b2), 1e-5)

    def test_problem2_warm_start(self):
        np.testing.assert_array_almost_equal(np.round(self.r3.primal_soln, decimals=2),
                                             np.array([0.3, 0.7]))
        np.testing.assert_array_almost_equal(np.round(self.r3.dual_soln, decimals=1),
                                             np.array([4.2, 0., 0., 1.3, 0., 0.2]))

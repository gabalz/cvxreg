import unittest
import numpy as np
from scipy import sparse

from cvxreg.optim.socprog import (
    socp_solve, socp_nonnegative_cone, socp_second_order_cone,
    SocpOptimTask, SOCP_BACKEND__LBFGS, SOCP_BACKEND__CLARABEL,
)


def _make_problem0(backend):
    H = sparse.csc_matrix([[1., 1.], [1., 2.]])
    H = sparse.triu(H).tocsc()
    q = np.array([-1., -3.])
    A = sparse.csc_matrix([[-1., 0.]])
    b = np.array([0.])
    cones = [socp_nonnegative_cone(1, backend)]
    return H, q, A, b, cones


def _make_problem1(backend):
    H = sparse.csc_matrix([[1., 1.], [1., 2.]])
    H = sparse.triu(H).tocsc()
    q = np.array([-1., -3.])
    A = sparse.csc_matrix([[-1., 0.], [0., 0.], [1., 1.]])
    b = np.array([0., 1., 0.])
    cones = [socp_nonnegative_cone(1, backend), socp_second_order_cone(2, backend)]
    return H, q, A, b, cones


def _make_problem2(backend):
    H = sparse.block_diag([np.zeros((4, 4)), 2.0])
    H = sparse.triu(H).tocsc()
    q = np.array([1., 1., 1., 1., 0.])
    A = sparse.csc_matrix([
        [0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 1., 1., 0.],
    ])
    b = np.array([2., 0., 0., 0.])
    cones = [socp_second_order_cone(2, backend), socp_second_order_cone(2, backend)]
    return H, q, A, b, cones


class TestSocpLbfgsTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H0, q0, A0, b0, cones0 = _make_problem0(SOCP_BACKEND__LBFGS)
        cls.task0 = SocpOptimTask(H0, q0, A0, b0, cones0)
        cls.x0a = np.zeros_like(q0) + 0.01
        cls.x0b = np.array([-0.0025, 0.25])

    def test_fun_no_violation(self):
        self.assertAlmostEqual(self.task0.fun(self.x0a), -0.03975)

    def test_jac_no_violation(self):
        np.testing.assert_array_almost_equal(self.task0.jac(self.x0a), np.array([-0.98, -2.97]))

    def test_jac_finite_diff_no_violation(self):
        np.testing.assert_array_almost_equal(
            self.task0.jac_finite_difference(self.x0a), np.array([-0.98, -2.97]))

    def test_fun_ineq_violation(self):
        self.assertAlmostEqual(np.round(self.task0.fun(self.x0b), decimals=6), 2.439378)

    def test_jac_ineq_violation(self):
        np.testing.assert_array_almost_equal(
            np.round(self.task0.jac(self.x0b), decimals=6),
            np.array([-2500.7525, -2.5025]))

    def test_jac_finite_diff_ineq_violation(self):
        np.testing.assert_array_almost_equal(
            np.round(self.task0.jac_finite_difference(self.x0b), decimals=6),
            np.array([-2500.7525, -2.5025]))


class TestSocpLbfgsSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H0, q0, A0, b0, cones0 = _make_problem0(SOCP_BACKEND__LBFGS)
        cls.r0 = socp_solve(H0, q0, A0, b0, cones0, backend=SOCP_BACKEND__LBFGS, verbose=False)
        H1, q1, A1, b1, cones1 = _make_problem1(SOCP_BACKEND__LBFGS)
        cls.r1 = socp_solve(H1, q1, A1, b1, cones1, backend=SOCP_BACKEND__LBFGS, verbose=False)
        H2, q2, A2, b2, cones2 = _make_problem2(SOCP_BACKEND__LBFGS)
        cls.r2 = socp_solve(H2, q2, A2, b2, cones2, backend=SOCP_BACKEND__LBFGS, verbose=False)

    def test_problem0_solution(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r0.primal_soln, decimals=3), np.array([-0., 1.5]))

    def test_problem0_niterations(self):
        self.assertEqual(self.r0.niterations, 15)

    def test_problem0_stats(self):
        self.assertEqual(self.r0.solver_stats, ('Success',))

    def test_problem1_solution(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r1.primal_soln, decimals=3), np.array([-0., 1.]))

    def test_problem1_niterations(self):
        self.assertEqual(self.r1.niterations, 26)

    def test_problem2_solution(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r2.primal_soln, decimals=6),
            np.array([-1.0001, -1.0001, -0.2501, -0.2501, -0.5]))

    def test_problem2_niterations(self):
        self.assertEqual(self.r2.niterations, 28)


class TestSocpClarabelSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H0, q0, A0, b0, cones0 = _make_problem0(SOCP_BACKEND__CLARABEL)
        cls.r0 = socp_solve(H0, q0, A0, b0, cones0, backend=SOCP_BACKEND__CLARABEL, verbose=False)
        H1, q1, A1, b1, cones1 = _make_problem1(SOCP_BACKEND__CLARABEL)
        cls.r1 = socp_solve(H1, q1, A1, b1, cones1, backend=SOCP_BACKEND__CLARABEL, verbose=False)
        cls.x1 = cls.r1.primal_soln
        cls.r1b = socp_solve(H1, q1, A1, b1, cones1, x0=cls.x1,
                             backend=SOCP_BACKEND__CLARABEL, verbose=False)
        cls.r1c = socp_solve(H1, q1, A1, b1, cones1, x0=cls.x1 * 0.5,
                             backend=SOCP_BACKEND__CLARABEL, verbose=False)
        H2, q2, A2, b2, cones2 = _make_problem2(SOCP_BACKEND__CLARABEL)
        cls.r2 = socp_solve(H2, q2, A2, b2, cones2, backend=SOCP_BACKEND__CLARABEL, verbose=False)

    def test_problem0_solution(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r0.primal_soln, decimals=3), np.array([0., 1.5]))

    def test_problem0_dual(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r0.dual_soln, decimals=4), np.array([0.5]))

    def test_problem0_niterations(self):
        self.assertEqual(self.r0.niterations, 6)

    def test_problem1_solution(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r1.primal_soln, decimals=3), np.array([0., 1.]))

    def test_problem1_dual(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r1.dual_soln, decimals=4), np.array([1., 1., 1.]))

    def test_problem1_niterations(self):
        self.assertEqual(self.r1.niterations, 7)

    def test_problem1_stats(self):
        import clarabel
        self.assertEqual(self.r1.solver_stats[:1], (clarabel.SolverStatus.Solved,))

    def test_problem1_warm_start_from_solution(self):
        import clarabel
        np.testing.assert_array_almost_equal(
            np.round(self.r1b.primal_soln, decimals=3), np.array([0., 1.]))
        self.assertIsNone(self.r1b.dual_soln)
        self.assertEqual(self.r1b.solver_stats[:1], (clarabel.SolverStatus.Solved,))

    def test_problem1_warm_start_from_half_solution(self):
        import clarabel
        np.testing.assert_array_almost_equal(
            np.round(self.r1c.primal_soln, decimals=3), np.array([0., 1.]))
        self.assertIsNone(self.r1c.dual_soln)
        self.assertEqual(self.r1c.solver_stats[:1], (clarabel.SolverStatus.Solved,))

    def test_problem2_solution(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r2.primal_soln, decimals=6),
            np.array([-1., -1., -0.25, -0.25, -0.5]))

    def test_problem2_dual(self):
        np.testing.assert_array_almost_equal(
            np.round(self.r2.dual_soln, decimals=6),
            np.array([1., -1., 1., -1.]))

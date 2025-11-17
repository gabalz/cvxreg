import numpy as np
from scipy import sparse

from ai.gandg.optim.optim_task import OptimTask


SOCP_BACKEND__CLARABEL = 'clarabel'
SOCP_BACKEND__LBFGS = 'L-BFGS'
SOCP_BACKEND__DEFAULT = SOCP_BACKEND__CLARABEL


def convert_matrix_to_socp_solver_format(x, backend):
    if x.getformat() == 'csc':
        return x
    if backend in (SOCP_BACKEND__CLARABEL,):
        y = sparse.csc_matrix(x)
    else:
        raise NotImplementedError('Not supported backend: {}'.format(backend))
    return y


def socp_nonnegative_cone(count, backend):
    if backend == SOCP_BACKEND__CLARABEL:
        import clarabel
        return clarabel.NonnegativeConeT(count)
    else:
        return ('ineq', count)
    raise NotImplementedError('Not supported backend: {}'.format(backend))    


def socp_second_order_cone(count, backend):
    if backend == SOCP_BACKEND__CLARABEL:
        import clarabel
        return clarabel.SecondOrderConeT(count)
    else:
        return ('soc', count)
    raise NotImplementedError('Not supported backend: {}'.format(backend))


class SOCPResult:
    """"Result and statistics of a SOCP optimization."""
    def __init__(self, primal_soln, dual_soln, niterations, solver_stats):
        """
        :param primal_soln: primal solution vector
        :param dual_soln: dual solution vector
        :param stat: statistics
        :return: QPResult object
        """
        self.primal_soln = primal_soln
        self.dual_soln = dual_soln
        self.niterations = niterations
        self.solver_stats = solver_stats


def _check_socp_zero_feas(b, cones, backend):
    idx_ineq = []
    idx_soc = []
    idx = 0
    if backend == SOCP_BACKEND__CLARABEL:
        import clarabel
        for cone in cones:
            idx_next = idx + cone.dim
            idxs = list(range(idx, idx_next))
            if isinstance(cone, clarabel.NonnegativeConeT):
                idx_ineq += idxs
            if isinstance(cone, clarabel.SecondOrderConeT):
                idx_soc.append(idxs)
            idx = idx_next
        if len(idx_ineq) > 0:
            idx_ineq = np.array(idx_ineq)
            v_ineq = np.max(-b[idx_ineq])
            v_ineq_idx = idx_ineq[np.argmax(-b[idx_ineq])]
            v_ineq = -b[v_ineq_idx]
            assert v_ineq < 1e-6, f'Not feasible x0, v_ineq: {v_ineq:.6f}, @idx: {v_ineq_idx}!'
        if len(idx_soc) > 0:
            for idx in idx_soc:
                v_soc = np.linalg.norm(b[idx[1:]]) - b[idx[0]]
                assert v_soc < 1e-6, f'Not feasible x0, v_soc: {v_soc}, @idx: {idx}!'
    else:
        raise NotImplementedError('Not supported backend: {}'.format(backend))


def socp_solve(
    H, g, A, b, cones,
    x0=None, maxiter='AUTO',
    backend=SOCP_BACKEND__DEFAULT,
    lbfgs_memsize=10,
    verbose=True,
):
    """Solving second order cone programming (SOCP) problem of the form: 

        0.5*x'*H*x + g'*x, s.t. A*x + s = b, s in K.

        K>=0 = {s : s >= 0}
        Ksoc = {(s0, s1:) : ||s1:|| <= s0}
        Ksoc in other words: ||(b-Ax)[1:]|| <= (b-Ax)[0]

    :param H: objective symmetric positive semi-definite matrix
    :param g: objective vector
    :param A: constraint matrix
    :param b: constraint vector
    :param cones: list of cones
    :param x0: starting point
    :param maxiter: maximum number of iterations
    :param backend: quadratic programming solver
    :param verbose: whether to print verbose output
    :return: SOCPResult object representing the results and the statistics
    """
    niterations = 0
    solver_stats = None
    if backend == SOCP_BACKEND__CLARABEL:
        if maxiter == 'AUTO':
            maxiter = 500
        import clarabel
        settings = clarabel.DefaultSettings()
        settings.verbose = (verbose > 0)
        settings.max_iter = maxiter
        if x0 is not None:
            b = b - A.dot(x0)
            _check_socp_zero_feas(b, cones, backend)
            g = g + sparse.triu(H).dot(x0) + sparse.triu(H, 1).dot(x0)
        solver = clarabel.DefaultSolver(H, g, A, b, cones, settings)
        result = solver.solve()
        if verbose:
            print(f'status: {result.status}')
            if result.x is not None:
                print('max(A*x-b): {}'.format((A.dot(result.x)-b).max()))
        assert result.status in (clarabel.SolverStatus.Solved,
                                 clarabel.SolverStatus.AlmostSolved,
                                 clarabel.SolverStatus.MaxIterations,
                                 clarabel.SolverStatus.InsufficientProgress,
                                 clarabel.SolverStatus.MaxTime), f'status: {result.status}'
        primal_soln = np.array(result.x)
        dual_soln = np.array(result.z)
        niterations = result.iterations
        solver_stats = (result.status, result.obj_val, result.solve_time)
        if x0 is not None:
            primal_soln += x0
            dual_soln = None
    elif backend == SOCP_BACKEND__LBFGS:
        if maxiter == 'AUTO':
            maxiter = 10000

        task = SocpOptimTask(H, g, A, b, cones, verbose=verbose)
        if x0 is None:
            x0 = np.zeros_like(g)
        from ai.gandg.optim.lbfgs import LBFGS
        result = LBFGS(max_memory=lbfgs_memsize,
                       curve_tol=1e-15,
                       max_iter=maxiter).minimize(task, x0)
        primal_soln = result.x
        dual_soln = None
        niterations = result.niter
        solver_stats = (result.status,)
    else:
        raise NotImplementedError(f'Not supported backend: {backend}!')

    return SOCPResult(primal_soln, dual_soln, niterations, solver_stats)


class SocpOptimTask(OptimTask):
    def __init__(
        self,
        H: sparse.csc_matrix, g: np.ndarray,
        A: sparse.csc_matrix, b: np.ndarray,
        cones: list[tuple[str, int]],
        mu_ineq=1e6, mu_soc=1e4, # constraint penalty parameters
        verbose=0,
    ):
        OptimTask.__init__(self, verbose)
        Aineq = []
        bineq = []
        Asoc = []
        bsoc = []
        idx = 0
        for cone in cones:
            idx_next = idx + cone[1]
            if cone[0] == 'ineq':
                _A = Aineq
                _b = bineq
            elif cone[0] == 'soc':
                _A = Asoc
                _b = bsoc
            else:
                raise NotImplementedError(f'Not supported cone: {cone}!')
            _A.append(A[idx:idx_next, :])
            _b.append(b[idx:idx_next])
            idx = idx_next
        self.H = H
        self.Hdiag = H.diagonal()
        self.g = g
        if len(Aineq) > 0:
            self.Aineq = sparse.vstack(Aineq)
            self.bineq = np.concatenate(bineq)
            # row scaling to improve conditioning
            norms = np.sqrt(self.Aineq.multiply(self.Aineq).sum(axis=1).A1)
            np.maximum(norms, 1e-8, out=norms)
            self.Aineq = sparse.diags(1.0 / norms).dot(self.Aineq)
            self.bineq /= norms
        else:
            self.Aineq = np.empty((0, len(g)))
            self.bineq = np.empty(0)

        self.Asoc = Asoc
        self.AsocSq = [M[1:,:].T.dot(M[1:, :]) for M in Asoc]
        self.AsocTb = [M[1:,:].T.dot(b[1:]) for M, b in zip(Asoc, bsoc)]
        self.Asoc0 = [M[0, :].toarray().ravel() for M in Asoc]
        self.bsoc = bsoc

        self.mu_ineq = mu_ineq
        self.half_mu_ineq = 0.5 * mu_ineq
        self.mu_soc = mu_soc
        self.half_mu_soc = 0.5 * mu_soc

    def _cache(self, x: np.ndarray):
        self.cache_key = x
        self.Hx = self.H.dot(x) + self.H.T.dot(x) - self.Hdiag*x
        self.ineq_viols = self.Aineq.dot(x)
        self.ineq_viols -= self.bineq
        np.maximum(self.ineq_viols, 0.0, out=self.ineq_viols)
        self.soc_vals = []
        self.soc_norms = []
        for Asoc, bsoc in zip(self.Asoc, self.bsoc):
            v = Asoc.dot(x)
            v -= bsoc
            self.soc_vals.append(v[0])
            self.soc_norms.append(np.linalg.norm(v[1:]))
        self.soc_vals = np.array(self.soc_vals)
        self.soc_norms = np.array(self.soc_norms)
        self.soc_viols = self.soc_vals + self.soc_norms
        np.maximum(self.soc_viols, 0.0, out=self.soc_viols)

    def fun(self, x: np.ndarray):
        super().fun(x)
        self._cache(x)
        fval = (0.5 * x.dot(self.Hx)) + self.g.dot(x)
        mineq = len(self.bineq)
        if mineq > 0:
            fval += (self.half_mu_ineq / mineq) * np.sum(np.square(self.ineq_viols))
        msoc = len(self.bsoc)
        if msoc > 0:
            fval += (self.half_mu_soc / msoc) * np.sum(np.square(self.soc_viols))
        return fval

    def jac(self, x: np.ndarray):
        super().jac(x)
        if not np.array_equiv(self.cache_key, x):
            self._cache(x)
        grad = self.Hx + self.g
        mineq = len(self.bineq)
        if mineq > 0:
            grad += (self.mu_ineq / mineq) * self.Aineq.T.dot(self.ineq_viols)
        msoc = len(self.bsoc)
        if msoc > 0:
            adj_mu = self.mu_soc / msoc
            for k in np.where(self.soc_viols > 1e-6)[0]:
                v = self.AsocSq[k].dot(x)
                v -= self.AsocTb[k]
                v /= (self.soc_norms[k] + 1e-8)
                v += self.Asoc0[k]
                v *= adj_mu * self.soc_viols[k]
                grad += v
        return grad


def _socp_lbfgs_test():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> backend = SOCP_BACKEND__LBFGS

    # #-----------------------------------------------
    >>> H0 = sparse.csc_matrix([[1., 1.], [1., 2.]])
    >>> H0 = sparse.triu(H0).tocsc()
    >>> q0 = np.array([-1., -3.])
    >>> A0 = sparse.csc_matrix([[-1., 0.]])
    >>> b0 = np.array([0.])
    >>> cones0 = [socp_nonnegative_cone(1, backend)]

    >>> task0 = SocpOptimTask(H0, q0, A0, b0, cones0)

    >>> x0a = np.zeros_like(q0) + 0.01  # no cons. viol.
    >>> task0.fun(x0a)
    -0.03975
    >>> task0.jac(x0a)
    array([-0.98, -2.97])
    >>> task0.jac_finite_difference(x0a)
    array([-0.98, -2.97])

    >>> x0b = np.array([-0.0025, 0.25])  # ineq. cons. viol.
    >>> np.round(task0.fun(x0b), decimals=6)
    2.439378
    >>> np.round(task0.jac(x0b), decimals=6)
    array([-2500.7525,    -2.5025])
    >>> np.round(task0.jac_finite_difference(x0b), decimals=6)
    array([-2500.7525,    -2.5025])

    >>> x0c = np.array([1.000e-03, 1.464e+00])
    >>> np.round(task0.fun(x0c), decimals=6)
    -2.24824
    >>> np.round(task0.jac(x0c), decimals=6)
    array([ 0.465, -0.071])
    >>> np.round(task0.jac_finite_difference(x0c), decimals=6)
    array([ 0.465, -0.071])

    >>> r0 = socp_solve(H0, q0, A0, b0, cones0, backend=backend, verbose=False)
    >>> x0 = r0.primal_soln
    >>> np.round(x0, decimals=3)
    array([-0. ,  1.5])
    >>> r0.niterations
    15
    >>> r0.solver_stats
    ('Success',)

    #-----------------------------------------------
    >>> H1 = sparse.csc_matrix([[1., 1.], [1., 2.]])
    >>> H1 = sparse.triu(H1).tocsc()
    >>> q1 = np.array([-1., -3.])
    >>> A1 = sparse.csc_matrix([[-1., 0.],
    ...                         [0., 0.],
    ...                         [1., 1.]])
    >>> b1 = np.array([0., 1., 0.])
    >>> cones1 = [socp_nonnegative_cone(1, backend),
    ...           socp_second_order_cone(2, backend)]  # |x| <= 1

    >>> task1 = SocpOptimTask(H1, q1, A1, b1, cones1)

    >>> x1a = np.zeros_like(q1) + 0.01  # no cons. viol.
    >>> task1.fun(x1a)
    -0.03975
    >>> task1.jac(x1a)
    array([-0.98, -2.97])
    >>> task1.jac_finite_difference(x1a)
    array([-0.98, -2.97])

    >>> x1b = np.array([-0.0025, 0.25])  # ineq. cons. viol.
    >>> np.round(task1.fun(x1b), decimals=6)
    2.439378
    >>> np.round(task1.jac(x1b), decimals=6)
    array([-2500.7525,    -2.5025])
    >>> np.round(task1.jac_finite_difference(x1b), decimals=6)
    array([-2500.7525,    -2.5025])

    >>> x1c = np.array([np.sqrt(0.005), np.sqrt(0.9075)])  # soc. cons. viol.
    >>> np.round(task1.fun(x1c), decimals=6)
    0.772223
    >>> np.round(task1.jac(x1c), decimals=6)
    array([233.409559, 232.362187])
    >>> np.round(task1.jac_finite_difference(x1c), decimals=6)
    array([233.409561, 232.362189])

    >>> r1 = socp_solve(H1, q1, A1, b1, cones1, backend=backend, verbose=False)
    >>> x1 = r1.primal_soln
    >>> np.round(x1, decimals=3)
    array([-0.,  1.])
    >>> r1.niterations
    26
    >>> r1.solver_stats
    ('Success',)

    -------------------------------------------------
    >>> H2 = sparse.block_diag([np.zeros((4,4)), 2.0])
    >>> H2 = sparse.triu(H2).tocsc()
    >>> q2 = np.array([1., 1., 1., 1., 0.])
    >>> A2 = sparse.csc_matrix([
    ...     [0., 0., 0., 0., 0.],
    ...     [1., 1., 0., 0., 0.],
    ...     [0., 0., 0., 0., 1.],
    ...     [0., 0., 1., 1., 0.],
    ... ])
    >>> b2 = np.array([2., 0., 0., 0.])
    >>> cones2 = [socp_second_order_cone(2, backend),
    ...           socp_second_order_cone(2, backend)]

    >>> task2 = SocpOptimTask(H2, q2, A2, b2, cones2)
    >>> x2a = np.zeros(5)
    >>> np.round(task2.fun(x2a), decimals=6)
    0.0
    >>> np.round(task2.jac(x2a), decimals=6)
    array([1., 1., 1., 1., 0.])
    >>> np.round(task2.jac_finite_difference(x2a, eps=1e-8), decimals=6)
    array([1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.3e-05])

    >>> x2b = np.array([0.0, 0.0, 0.0, 0.0, -0.01])
    >>> np.round(task2.fun(x2b), decimals=6)
    0.0001
    >>> np.round(task2.jac(x2b), decimals=6)
    array([ 1.  ,  1.  ,  1.  ,  1.  , -0.02])
    >>> np.round(task2.jac_finite_difference(x2b, eps=1e-8), decimals=6)
    array([ 1.  ,  1.  ,  1.  ,  1.  , -0.02])

    >>> x2c = np.array([0.0, 0.0, 0.0, 0.0, 0.01])
    >>> np.round(task2.fun(x2c), decimals=6)
    0.2501
    >>> np.round(task2.jac(x2c), decimals=6)
    array([ 1.  ,  1.  ,  1.  ,  1.  , 50.02])
    >>> np.round(task2.jac_finite_difference(x2c, eps=1e-8), decimals=6)
    array([ 1.  ,  1.  ,  1.  ,  1.  , 50.02])

    >>> r2 = socp_solve(H2, q2, A2, b2, cones2, backend=backend, verbose=False)
    >>> x2 = r2.primal_soln
    >>> np.round(x2, decimals=6)
    array([-1.0001, -1.0001, -0.2501, -0.2501, -0.5   ])
    >>> r2.niterations
    28
    >>> r2.solver_stats
    ('Success',)
    """
    pass


def _socp_clarabel_test():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> backend = SOCP_BACKEND__CLARABEL

    #-----------------------------------------------
    >>> H0 = sparse.csc_matrix([[1., 1.], [1., 2.]])
    >>> H0 = sparse.triu(H0).tocsc()
    >>> q0 = np.array([-1., -3.])
    >>> A0 = sparse.csc_matrix([[-1., 0.]])
    >>> b0 = np.array([0.])
    >>> cones0 = [socp_nonnegative_cone(1, backend)]
    >>> r0 = socp_solve(H0, q0, A0, b0, cones0, backend=backend, verbose=False)
    >>> x0 = r0.primal_soln
    >>> np.round(x0, decimals=3)
    array([0. , 1.5])
    >>> np.round(r0.dual_soln, decimals=4)
    array([0.5])
    >>> r0.niterations
    6
    >>> r0.solver_stats[:1]
    (Solved,)

    #-----------------------------------------------
    >>> H1 = sparse.csc_matrix([[1., 1.], [1., 2.]])
    >>> H1 = sparse.triu(H1).tocsc()
    >>> q1 = np.array([-1., -3.])
    >>> A1 = sparse.csc_matrix([[-1., 0.],
    ...                         [0., 0.],
    ...                         [1., 1.]])
    >>> b1 = np.array([0., 1., 0.])
    >>> cones1 = [socp_nonnegative_cone(1, backend),
    ...           socp_second_order_cone(2, backend)]  # |x| <= 1
    >>> r1 = socp_solve(H1, q1, A1, b1, cones1, backend=backend, verbose=False)
    >>> x1 = r1.primal_soln
    >>> np.round(x1, decimals=3)
    array([0., 1.])
    >>> np.round(r1.dual_soln, decimals=4)
    array([1., 1., 1.])
    >>> r1.niterations
    7
    >>> r1.solver_stats[:1]
    (Solved,)

    >>> r1b = socp_solve(H1, q1, A1, b1, cones1, x0=x1, backend=backend, verbose=False)
    >>> x1b = r1b.primal_soln
    >>> np.round(x1b, decimals=3)
    array([0., 1.])
    >>> r1b.dual_soln is None
    True
    >>> r1b.niterations
    6
    >>> r1b.solver_stats[:1]
    (Solved,)

    >>> r1c = socp_solve(H1, q1, A1, b1, cones1, x0=x1*0.5, backend=backend, verbose=False)
    >>> x1c = r1c.primal_soln
    >>> np.round(x1c, decimals=3)
    array([0., 1.])
    >>> r1c.dual_soln is None
    True
    >>> r1c.niterations
    6
    >>> r1c.solver_stats[:1]
    (Solved,)

    #-----------------------------------------------
    >>> H2 = sparse.block_diag([np.zeros((4,4)), 2.0])
    >>> H2 = sparse.triu(H2).tocsc()
    >>> q2 = np.array([1., 1., 1., 1., 0.])
    >>> A2 = sparse.csc_matrix([
    ...     [0., 0., 0., 0., 0.],
    ...     [1., 1., 0., 0., 0.],
    ...     [0., 0., 0., 0., 1.],
    ...     [0., 0., 1., 1., 0.],
    ... ])
    >>> b2 = np.array([2., 0., 0., 0.])
    >>> cones2 = [socp_second_order_cone(2, backend),
    ...           socp_second_order_cone(2, backend)]
    >>> r2 = socp_solve(H2, q2, A2, b2, cones2, backend=backend, verbose=False)
    >>> x2 = r2.primal_soln
    >>> np.round(x2, decimals=6)
    array([-1.  , -1.  , -0.25, -0.25, -0.5 ])
    >>> np.round(r2.dual_soln, decimals=6)
    array([ 1., -1.,  1., -1.])
    """
    pass

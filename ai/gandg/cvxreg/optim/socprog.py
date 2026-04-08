import numpy as np
from scipy import sparse

from ai.gandg.cvxreg.optim.optim_task import OptimTask


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
        from ai.gandg.cvxreg.optim.lbfgs import LBFGS
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
        mu_ineq=1e6, mu_soc=1e4,  # constraint penalty parameters
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
        self.AsocSq = [M[1:, :].T.dot(M[1:, :]) for M in Asoc]
        self.AsocTb = [M[1:, :].T.dot(b[1:]) for M, b in zip(Asoc, bsoc)]
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



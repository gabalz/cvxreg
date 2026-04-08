import numpy as np
from scipy import sparse


QP_BACKEND__OSQP = 'osqp'
QP_BACKEND__CLARABEL = 'clarabel'
QP_BACKEND__DEFAULT = QP_BACKEND__CLARABEL


def convert_matrix_to_qp_solver_format(x, backend):
    if backend in (QP_BACKEND__OSQP, QP_BACKEND__CLARABEL):
        y = sparse.csc_matrix(x)
    else:
        raise NotImplementedError('Not supported backend: {}'.format(backend))
    return y


class QPResult:
    """"Result and statistics of a QP optimization."""
    def __init__(self, primal_soln, dual_soln, stat):
        """
        :param primal_soln: primal solution vector
        :param dual_soln: dual solution vector
        :param stat: statistics
        :return: QPResult object
        """
        self.primal_soln = primal_soln
        self.dual_soln = dual_soln
        self.stat = stat


def _qp_solve_osqp(H, g, A, lb, ub, x0, y0, maxiter, verbose):
    if maxiter == 'AUTO':
        maxiter = 10000
    import osqp
    m = osqp.OSQP()
    H = sparse.csc_matrix(H)
    A = sparse.csc_matrix(A)
    if lb is None:
        lb = -np.inf*np.ones(ub.shape)
    m.setup(
        P=H, q=g, A=A, l=lb, u=ub,
        max_iter=maxiter, verbose=verbose,
        polishing=True,
    )
    if x0 is not None:
        m.warm_start(x=x0, y=y0)
    result = m.solve(raise_error=True)
    if verbose:
        print('status({}): {}'.format(result.info.status_val, result.info.status))
        if result.x is not None:
            print('max(A*x-b): {}'.format((A.dot(result.x)-ub).max()))
    assert result.info.status_val in (-2, 1, 2), \
        'status ({}): {}'.format(result.info.status_val, result.info.status)
    primal_soln = result.x
    dual_soln = result.y
    return primal_soln, dual_soln


def _qp_solve_clarabel(H, g, A, lb, ub, x0, y0, maxiter, verbose):
    assert lb is None, 'lb is not supported for clarabel!'
    if maxiter == 'AUTO':
        maxiter = 500
    import clarabel
    settings = clarabel.DefaultSettings()
    settings.verbose = (verbose > 0)
    settings.max_iter = maxiter
    cones = [clarabel.NonnegativeConeT(len(ub))]
    solver = clarabel.DefaultSolver(H, g, A, ub, cones, settings)
    result = solver.solve()
    if verbose:
        print(f'status: {result.status}')
        if result.x is not None:
            print('max(A*x-ub): {}'.format((A.dot(result.x)-ub).max()))
    stat = result.status
    assert result.status in (clarabel.SolverStatus.Solved,
                             clarabel.SolverStatus.AlmostSolved,
                             clarabel.SolverStatus.MaxIterations,
                             clarabel.SolverStatus.InsufficientProgress,
                             clarabel.SolverStatus.MaxTime), f'status: {result.status}'
    primal_soln = np.array(result.x)
    dual_soln = np.array(result.z)
    return primal_soln, dual_soln, stat


def qp_solve(
    H, g, A, ub, lb=None,
    x0=None, y0=None,
    maxiter='AUTO',
    backend=QP_BACKEND__DEFAULT,
    verbose=True,
):
    """Solving quadratic programming (QP) problem of the form:

        0.5*x'*H*x + g'*x, s.t. lb <= A*x <= ub.

    :param H: objective symmetric positive semi-definite matrix
    :param g: objective vector
    :param A: constraint matrix
    :param ub: constraint vector
    :param lb: constraint vector
    :param x0: warm starting primal initialization vector
    :param y0: warm starting dual initialization vector
    :param maxiter: maximum number of iterations
    :param backend: quadratic programming solver
    :param verbose: whether to print verbose output
    :return: QPResult object representing the results and the statistics
    """
    H = H.astype(float, copy=False)
    g = g.astype(float, copy=False)
    A = A.astype(float, copy=False)
    if lb is not None:
        lb = lb.astype(float, copy=False)
    if ub is not None:
        ub = ub.astype(float, copy=False)
    stat = None
    if backend == QP_BACKEND__OSQP:
        primal_soln, dual_soln = _qp_solve_osqp(
            H, g, A, lb, ub, x0, y0, maxiter, verbose)
    elif backend == QP_BACKEND__CLARABEL:
        primal_soln, dual_soln, stat = _qp_solve_clarabel(
            H, g, A, lb, ub, x0, y0, maxiter, verbose)
    else:
        raise NotImplementedError('Not supported backend: {}'.format(backend))

    return QPResult(primal_soln, dual_soln, stat)



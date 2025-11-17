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
    x = x0
    stat = None
    if backend == QP_BACKEND__OSQP:
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
        if x is not None:
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
    elif backend == QP_BACKEND__CLARABEL:
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
    else:
        raise NotImplementedError('Not supported backend: {}'.format(backend))

    return QPResult(primal_soln, dual_soln, stat)


def _quadprog_osqp_test():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> backend = QP_BACKEND__OSQP

    >>> H1 = np.array([[3., 1.], [1., 1.]]) * 2.0
    >>> g1 = np.array([1., 6.])
    >>> A1 = np.array([[-2., -3.], [-1., 0.], [0., -1.]])
    >>> b1 = np.array([-4., 0., 0.])
    >>> r1 = qp_solve(H1, g1, A1, b1, backend=backend, verbose=False)
    >>> x1 = r1.primal_soln
    >>> np.round(x1, decimals=6)
    array([0.5, 1. ])
    >>> np.round(r1.dual_soln, decimals=6)
    array([3., 0., 0.])
    >>> np.round(0.5 * x1.dot(H1.dot(x1)) + g1.dot(x1), decimals=6)
    9.25
    >>> np.round(np.max(A1.dot(x1) - b1), decimals=6)
    0.0

    >>> H2 = sparse.csc_matrix([[4, 1], [1, 2]])
    >>> g2 = np.array([1, 1])
    >>> A2 = sparse.csc_matrix(
    ...     [[-1, -1], [-1, 0], [0, -1],
    ...      [1, 1], [1, 0], [0, 1]],
    ... )
    >>> b2 = np.array([-1, 0, 0, 1, 0.7, 0.7])
    >>> r2 = qp_solve(H2, g2, A2, b2, backend=backend, verbose=False)
    >>> x2 = r2.primal_soln
    >>> np.round(x2, decimals=2)
    array([0.3, 0.7])
    >>> np.round(r2.dual_soln + 1e-7, decimals=1)
    array([2.9, 0. , 0. , 0. , 0. , 0.2])
    >>> np.round(0.5*x2.dot(H2.dot(x2)) + g2.dot(x2), decimals=2)
    1.88
    >>> max(np.max(A2.dot(x2) - b2), 1e-5)
    1e-05

    >>> x0 = np.array([0.2, 0.5])
    >>> y0 = np.array([2.8, 0.0, 0.0, 0.0, 0.0, 0.3])
    >>> r3 = qp_solve(H2, g2, A2, b2, x0=x0, y0=y0, backend=backend, verbose=False)
    >>> np.round(r3.primal_soln, decimals=2)
    array([0.3, 0.7])
    >>> np.round(r3.dual_soln, decimals=1)
    array([2.9, 0. , 0. , 0. , 0. , 0.2])
    """
    pass


def _quadprog_clarabel_test():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> backend = QP_BACKEND__CLARABEL

    >>> H1 = np.array([[3., 1.], [1., 1.]]) * 2.0
    >>> g1 = np.array([1., 6.])
    >>> A1 = np.array([[-2., -3.], [-1., 0.], [0., -1.]])
    >>> b1 = np.array([-4., 0., 0.])
    >>> H1 = convert_matrix_to_qp_solver_format(H1, backend)
    >>> A1 = convert_matrix_to_qp_solver_format(A1, backend)
    >>> r1 = qp_solve(H1, g1, A1, b1, backend=backend, verbose=False)
    >>> x1 = r1.primal_soln
    >>> np.round(x1, decimals=6)
    array([0.5, 1. ])
    >>> np.round(r1.dual_soln, decimals=6)
    array([3., 0., 0.])
    >>> np.round(0.5 * x1.dot(H1.dot(x1)) + g1.dot(x1), decimals=6)
    9.25
    >>> np.round(np.max(A1.dot(x1) - b1), decimals=6)
    -0.0

    >>> H2 = sparse.csc_matrix([[4, 1], [1, 2]])
    >>> g2 = np.array([1, 1])
    >>> A2 = sparse.csc_matrix(
    ...     [[-1, -1], [-1, 0], [0, -1],
    ...      [1, 1], [1, 0], [0, 1]],
    ... )
    >>> b2 = np.array([-1, 0, 0, 1, 0.7, 0.7])
    >>> r2 = qp_solve(H2, g2, A2, b2, backend=backend, verbose=False)
    >>> x2 = r2.primal_soln
    >>> np.round(x2, decimals=2)
    array([0.3, 0.7])
    >>> np.round(r2.dual_soln + 1e-7, decimals=1)
    array([4.2, 0. , 0. , 1.3, 0. , 0.2])
    >>> np.round(0.5*x2.dot(H2.dot(x2)) + g2.dot(x2), decimals=2)
    1.88
    >>> max(np.max(A2.dot(x2) - b2), 1e-5)
    1e-05

    >>> x0 = np.array([0.2, 0.5])
    >>> y0 = np.array([2.8, 0.0, 0.0, 0.0, 0.0, 0.3])
    >>> r3 = qp_solve(H2, g2, A2, b2, x0=x0, y0=y0, backend=backend, verbose=False)
    >>> np.round(r3.primal_soln, decimals=2)
    array([0.3, 0.7])
    >>> np.round(r3.dual_soln, decimals=1)
    array([4.2, 0. , 0. , 1.3, 0. , 0.2])
    """
    pass

import numpy as np

from scipy import sparse


SOCP_BACKEND__CLARABEL = 'clarabel'
SOCP_BACKEND__DEFAULT = SOCP_BACKEND__CLARABEL


def convert_matrix_to_socp_solver_format(x, backend):
    if backend in (SOCP_BACKEND__CLARABEL,):
        y = sparse.csc_matrix(x)
    else:
        raise NotImplementedError('Not supported backend: {}'.format(backend))
    return y


def socp_nonnegative_cone(count, backend):
    if backend == SOCP_BACKEND__CLARABEL:
        import clarabel
        return clarabel.NonnegativeConeT(count)
    raise NotImplementedError('Not supported backend: {}'.format(backend))    


def socp_second_order_cone(count, backend):
    if backend == SOCP_BACKEND__CLARABEL:
        import clarabel
        return clarabel.SecondOrderConeT(count)
    raise NotImplementedError('Not supported backend: {}'.format(backend))


class SOCPResult:
    """"Result and statistics of a SOCP optimization."""
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


def socp_solve(
    H, g, A, b, cones,
    maxiter='AUTO',
    backend=SOCP_BACKEND__DEFAULT,
    verbose=True,
):
    """Solving second order cone programming (SOCP) problem of the form: 

        0.5*x'*H*x + g'*x, s.t. A*x + s = b, s in K.

    :param H: objective symmetric positive semi-definite matrix
    :param g: objective vector
    :param A: constraint matrix
    :param b: constraint vector
    :param cones: list of cones
    :param maxiter: maximum number of iterations
    :param backend: quadratic programming solver
    :param verbose: whether to print verbose output
    :return: SOCPResult object representing the results and the statistics
    """
    stat = None
    if backend == SOCP_BACKEND__CLARABEL:
        if maxiter == 'AUTO':
            maxiter = 500
        import clarabel
        settings = clarabel.DefaultSettings()
        settings.verbose = (verbose > 0)
        settings.max_iter = maxiter
        solver = clarabel.DefaultSolver(H, g, A, b, cones, settings)
        result = solver.solve()
        if verbose:
            print(f'status: {result.status}')
            if result.x is not None:
                print('max(A*x-b): {}'.format((A.dot(result.x)-b).max()))
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

    return SOCPResult(primal_soln, dual_soln, stat)


def socp_clarabel_test():
    """
    >>> backend = SOCP_BACKEND__CLARABEL

    >>> H1 = sparse.csc_matrix([[1., 1.], [1., 2.]])
    >>> H1 = sparse.triu(H1).tocsc()
    >>> q1 = np.array([-1., -2.])
    >>> A1 = sparse.csc_matrix([[-1., 0.],
    ...                         [0., 0.],
    ...                         [1., 1.]])
    >>> b1 = np.array([0., 1., 0.])
    >>> cones1 = [socp_nonnegative_cone(1, backend),
    ...           socp_second_order_cone(2, backend)]
    >>> r1 = socp_solve(H1, q1, A1, b1, cones1, backend=backend, verbose=False)
    >>> x1 = r1.primal_soln
    >>> np.round(x1, decimals=3)
    array([0., 1.])
    >>> np.round(r1.dual_soln, decimals=4)
    array([0.0001, 0.0001, 0.0001])

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

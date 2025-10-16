
import numpy as np
import numba

from collections import namedtuple
from optim.optim_task import OptimTask
# from common.util import eprint


LineSearchResult = namedtuple(
    'LineSearchResult',
    ['step', 'x', 'fval'],
)


class BackTrackingLineSearch:
    def __init__(self, step_scale=0.5, min_step=1e-16, c_tol=1e-4):
        self.step_scale = step_scale
        self.min_step = min_step
        self.c_tol = c_tol

    def search(self, fun, fval0, x0, g0, p, step0=1.0):
        ftol = self.c_tol * p.dot(g0)
        step = step0
        x = x0 + (p if step == 1.0 else step * p)
        while True:
            if step < self.min_step:
                return None
            fval = fun(x)
            if (fval - fval0 < step * ftol):
                break
            step *= self.step_scale
            x[:] = p
            x *= step
            x += x0
        return LineSearchResult(step, x, fval)


LBFGSResult = namedtuple(
    'LBFGSResult',
    ['status', 'niter', 'nrestarts', 'x', 'fval', 'grad'],
)


class LBFGS:
    def __init__(self,
                 max_memory=5, max_iter=1000,
                 ls=BackTrackingLineSearch(),
                 ftol=0.0, ftol_patience=1,
                 acc_tol=1e-5, curve_tol=1e-8):
        self.max_memory = max_memory
        self.max_iter = max_iter
        self.ls = ls
        self.ftol = ftol
        self.ftol_patience = ftol_patience
        self.acc_tol = acc_tol
        self.curve_tol = curve_tol

    def minimize(self, task: OptimTask, x0: np.ndarray):
        mem_pos = 0
        mem_size = 0
        gamma = 1.0
        s_arr = np.zeros((len(x0), self.max_memory))
        y_arr = np.zeros((len(x0), self.max_memory))
        r_arr = np.zeros(self.max_memory)
        t_arr = np.zeros(self.max_memory)
        p = np.zeros_like(x0)

        x = x0.copy()
        fval = task.fun(x)
        assert np.array_equiv(x, x0), f'Argument of task.fun changes!'
        grad = task.jac(x)
        assert np.array_equiv(x, x0), f'Argument of task.jac changes!'

        xnorm = np.linalg.norm(x)
        gnorm = np.linalg.norm(grad)
        if gnorm < self.acc_tol * max(1.0, xnorm):
            return LBFGSResult('AlreadyMinimized', 0, 0, x, fval, grad)

        def _fun(x):
            return task.fun(x)

        step = 1.0 / max(gnorm, self.ls.min_step)
        niter = 0
        status = None
        ftol_patience = self.ftol_patience
        nrestarts = 0
        while True:
            if niter == self.max_iter:
                status = 'MaxIterReached'
                break
            niter += 1

            _calc_lbfgs_search_dir(p, grad, mem_pos, mem_size, gamma,
                                   self.max_memory, s_arr, y_arr, r_arr, t_arr)
            ls_res = self.ls.search(_fun, fval, x, grad, p, step0=step)
            if ls_res is None:
                if mem_size > 0:
                    nrestarts += 1
                    mem_pos = 0
                    mem_size = 0
                    gamma = 1.0
                    continue
                else:
                    status = 'LineSearchFailed'
                    break

            x_prev = x
            fval_prev = fval
            grad_prev = grad

            x = ls_res.x
            fval = ls_res.fval
            if fval_prev - fval < self.ftol:
                ftol_patience -= 1
                if ftol_patience == 0:
                    status = 'FvalProgressBelowTol'
                    break
            else:
                ftol_patience = self.ftol_patience
            # assert fval <= fval_prev, f'fval:{fval:.6f}, fval_prev:{fval_prev:.6f}'
            grad = task.jac(x)

            mem_pos += 1
            if mem_pos == self.max_memory:
                mem_pos = 0
            if mem_size < self.max_memory:
                mem_size += 1

            s_arr[:, mem_pos] = s = x - x_prev
            y_arr[:, mem_pos] = y = grad - grad_prev
            sTy = s.dot(y)
            if abs(sTy) < self.curve_tol:
                nrestarts += 1
                mem_pos = 0
                mem_size = 0
                gamma = 1.0
            else:
                r_arr[mem_pos] = 1.0 / sTy
                gamma = sTy / y.dot(y)

            xnorm = np.linalg.norm(x)
            gnorm = np.linalg.norm(grad)
            if gnorm < self.acc_tol * max(1.0, xnorm):
                status = 'Success'
                break
            step = 1.0

        return LBFGSResult(status, niter, nrestarts, x, fval, grad)


@numba.njit
def _calc_lbfgs_search_dir(p, grad, mem_pos, mem_size, gamma,
                           max_memory, s_arr, y_arr, r_arr, t_arr):
    p[:] = -grad
    i = mem_pos
    for _ in range(mem_size):
        i -= 1;
        if i < 0:
            i = max_memory - 1
        t_arr[i] = r_arr[i] * np.sum(p * s_arr[:, i])
        p -= t_arr[i] * y_arr[:, i]
    p *= gamma
    for _ in range(mem_size):
        beta = r_arr[i] * np.sum(p * y_arr[:, i])
        p += s_arr[:, i] * (t_arr[i] - beta)
        i += 1
        if i >= max_memory:
            i = 0
    return p


def _lbfgs_tests():
    """
    >>> np.set_printoptions(legacy='1.25')

    >>> class QuadFun(OptimTask):
    ...     def fun(self, x):
    ...         super().fun(x)
    ...         return 0.5 * x.dot(x)
    ...     def jac(self, x):
    ...         super().jac(x)
    ...         return x
    >>> res = LBFGS().minimize(QuadFun(), x0=np.ones(2))
    >>> res.status
    'Success'
    >>> res.x
    array([0., 0.])
    >>> res.niter
    2
    >>> res.nrestarts
    0

    >>> class Himmelblau(OptimTask):
    ...     def _cache(self, x):
    ...         self.cache_key = x
    ...         self.v1 = x[0]**2 + x[1] - 11.0
    ...         self.v2 = x[0] + x[1]**2 - 7.0
    ...     def fun(self, x):
    ...         super().fun(x)
    ...         self._cache(x)
    ...         return self.v1**2 + self.v2**2
    ...     def jac(self, x):
    ...         super().jac(x)
    ...         if not np.array_equiv(self.cache_key, x):
    ...             self._cache(x)
    ...         return np.array([
    ...             4.0 * self.v1 * x[0] + 2.0 * self.v2,
    ...             2.0 * self.v1 + 4.0 * self.v2 * x[1],
    ...         ])
    >>> res = LBFGS().minimize(Himmelblau(), x0=np.zeros(2))
    >>> res.status
    'Success'
    >>> np.round(res.x, decimals=5)
    array([3., 2.])
    >>> np.round(res.fval, decimals=6)
    0.0
    >>> res.niter
    11
    >>> res.nrestarts
    1
    """
    pass

import numpy as np

from ai.gandg.optim.optim_task import OptimTask


class SmoothDCFLocalOptimTask(OptimTask):
    """
    Nesterov smoothing: max_k v_k ~ mu * log ( sum_k exp(v_k / mu) ), k in [K],
    where mu is the smoothing constant (smooth_tol below).

    Smoothing is only applied to the gradient computation,
    its perturbation to the objective value is bounded by mu * log(K), and ignored.
    """
    def __init__(
        self, centers, phi, y,
        variant, is_convex, is_symmetrized,
        L_regularizer, L_regularizer_offset,
        L_sum_regularizer=0.0, verbose=0, con_penalty=1e6,
        smooth_tol=1e-6,
    ):
        OptimTask.__init__(self, verbose)
        self.centers = centers
        self.phi = phi
        self.y = y
        self.con_penalty = con_penalty
        self.variant = variant
        self.is_convex = is_convex
        self.is_symmetrized = is_symmetrized
        assert isinstance(is_symmetrized, bool)
        self.L_regularizer = L_regularizer
        self.L_regularizer_offset = L_regularizer_offset
        self.L_sum_regularizer = L_sum_regularizer
        self.smooth_tol = smooth_tol
        self.cache_key = None

    def _calc_expW_yhat(self, weights):
        from ai.gandg.algorithm.dcf.dcf import _dcf_calc_Xw
        v = _dcf_calc_Xw(self.phi, weights)
        yhat = np.max(v, axis=1)
        v -= yhat[:, None]
        v /= self.smooth_tol
        np.exp(v, out=v)
        v /= np.sum(v, axis=1)[:, None]
        return v, yhat

    def _cache(self, weights):
        n = len(self.y)
        K, d = self.centers.shape
        self.cache_key = weights
        weights = weights.reshape((K, -1), copy=True)

        if self.is_symmetrized:
            w1 = weights[:, 0::2]
            w2 = weights[:, 1::2]
            expW1, yhat1 = self._calc_expW_yhat(w1)
            expW2, yhat2 = self._calc_expW_yhat(w2)
            yhat = np.subtract(yhat1, yhat2, out=yhat1)
            resid = np.subtract(yhat, self.y, out=yhat)
            expW = (expW1, expW2)
            wnorms = np.concatenate([
                np.linalg.norm(w1[:, 1:], axis=1),
                np.linalg.norm(w2[:, 1:], axis=1),
            ])
        else:
            expW, yhat = self._calc_expW_yhat(weights)
            resid = np.subtract(yhat, self.y, out=yhat)
            wnorms = np.linalg.norm(weights[:, 1:], axis=1)

        wns, maxwne = _calc_logsumexp_maxwne(
            wnorms, self.smooth_tol, self.L_regularizer_offset)
        wsum = _calc_cvx_wsum(weights, d, self.is_convex, self.variant)
        self.cache = (resid, maxwne, wsum, wnorms, expW, wns)

    def fun(self, weights):
        super().fun(weights)
        self._cache(weights)
        resid, maxwne, wsum, wnorms = self.cache[:4]
        f = 0.5 * (resid.dot(resid)/len(resid) + self.L_regularizer * (maxwne ** 2))
        if wsum is not None:
            wsum = wsum.ravel()
            f += (0.5 * self.con_penalty) * wsum.dot(wsum)
        if self.L_sum_regularizer > 0.0:
            f += (0.5 * self.L_sum_regularizer) * wnorms.dot(wnorms)
        return f

    def jac(self, weights):
        super().jac(weights)
        n, K, nparams_pwf = self.phi.shape
        if self.cache_key is None or not np.array_equiv(self.cache_key, weights):
            self._cache(weights)
        resid, maxwne, wsum, wnorms, expW, wns = self.cache
        if self.is_symmetrized:
            grad = np.zeros(2*K*nparams_pwf)
            grad[0::2] = np.einsum('ik,ikj,i->kj', expW[0], self.phi, resid).ravel()
            grad[1::2] = -np.einsum('ik,ikj,i->kj', expW[1], self.phi, resid).ravel()
        else:
            grad = np.einsum('ik,ikj,i->kj', expW, self.phi, resid).ravel()
        grad /= n
        if maxwne > 0.0 and self.L_regularizer > 0.0:
            grad = _calc_logsumexp_maxwne_jac(
                grad, K, weights, wnorms, maxwne, wns,
                self.is_symmetrized, self.L_regularizer)
        if wsum is not None:
            d = self.centers.shape[1]
            grad += _calc_cvx_wsum_grad(wsum, self.con_penalty, K, d, self.variant)
        if self.L_sum_regularizer > 0.0:
            grad += _L_sum_reg_grad(weights, self.L_sum_regularizer,
                                    self.is_symmetrized, nparams_pwf)
        return grad


class NonSmoothDCFLocalOptimTask(OptimTask):
    def __init__(
        self, centers, phi, y,
        variant, is_convex, is_symmetrized,
        L_regularizer, L_regularizer_offset,
        L_sum_regularizer=0.0, verbose=0, con_penalty=1e6,
        noise_level=0.0
    ):
        super().__init__(verbose)
        self.centers = centers
        self.phi = phi
        self.y = y
        self.con_penalty = con_penalty
        self.variant = variant
        self.is_convex = is_convex
        self.is_symmetrized = is_symmetrized
        self.L_regularizer = L_regularizer
        self.L_regularizer_offset = L_regularizer_offset
        self.L_sum_regularizer = L_sum_regularizer
        self.noise_level = noise_level
        self.cache_key = np.zeros(0)
        self.cache = None
        K = self.centers.shape[0]
        self.one2K = np.arange(K)
        self.cache_key = None
    
    def _calc_maxk_yhat(self, weights, noise):
        from ai.gandg.algorithm.dcf.dcf import _dcf_calc_Xw
        Xw = _dcf_calc_Xw(self.phi, weights)
        maxk = np.argmax(Xw if noise is None else Xw+noise[None, :], axis=1)
        yhat = Xw[np.arange(Xw.shape[0]), maxk]
        return maxk, yhat

    def _cache(self, weights):
        noise = None
        n = len(self.y)
        K, d = self.centers.shape
        self.cache_key = weights
        weights = weights.reshape((K, -1), copy=True)

        if self.noise_level > 0.0:
            noise = np.random.randn(K) * self.noise_level
        if self.is_symmetrized:
            w1 = weights[:, 0::2]
            w2 = weights[:, 1::2]
            maxk1, yhat1 = self._calc_maxk_yhat(w1, noise)
            maxk2, yhat2 = self._calc_maxk_yhat(w2, noise)
            yhat = np.subtract(yhat1, yhat2, out=yhat1)
            resid = np.subtract(yhat, self.y, out=yhat)
            maxk = (maxk1, maxk2)
            weights = np.vstack([w1, w2])
        else:
            maxk, yhat = self._calc_maxk_yhat(weights, noise)
            resid = np.subtract(yhat, self.y, out=yhat)

        wnorms = np.linalg.norm(weights[:, 1:], axis=1)
        if self.noise_level > 0.0:
            noise = np.random.randn(len(wnorms)) * self.noise_level
        wnormk = np.argmax(wnorms if noise is None else wnorms + noise)
        max_wnorm = wnorms[wnormk]
        maxwne = max(0.0, max_wnorm - self.L_regularizer_offset)

        wsum = _calc_cvx_wsum(weights, d, self.is_convex, self.variant)
        self.cache = (resid, maxwne, wsum, wnorms, maxk, wnormk, weights[wnormk, 1:])

    def fun(self, weights):
        super().fun(weights)
        self._cache(weights)
        resid, maxwne, wsum, wnorms = self.cache[:4]
        f = 0.5 * (resid.dot(resid)/len(resid) + self.L_regularizer * (maxwne ** 2))
        if wsum is not None:
            wsum = wsum.ravel()
            f += (0.5 * self.con_penalty) * wsum.dot(wsum)
        if self.L_sum_regularizer > 0.0:
            f += (0.5 * self.L_sum_regularizer) * wnorms.dot(wnorms)
        return f

    def _jac_maxwne(self, grad, K, weights, maxwne, wmax, wnorms, wnormk, L_regularizer):
        nparams_pwf = len(wmax) + 1
        g = wmax * (maxwne * L_regularizer / wnorms[wnormk])
        if self.is_symmetrized:
            shift = int(wnormk >= K)
            scaled_nparams_pwf = 2 * nparams_pwf
            ind = (wnormk % K) * scaled_nparams_pwf
            grad[ind+2+shift:ind+scaled_nparams_pwf:2] += g
        else:
            ind = wnormk * nparams_pwf
            grad[ind+1:ind+nparams_pwf] += g
        return grad

    def jac(self, weights):
        super().jac(weights)
        n, K, nparams_pwf = self.phi.shape
        if self.cache_key is None or not np.array_equiv(self.cache_key, weights):
            self._cache(weights)
        resid, maxwne, wsum, wnorms, maxk, wnormk, wmax = self.cache
        if self.is_symmetrized:
            grad = np.zeros(2*K*nparams_pwf)
            mask = (maxk[0][:, None] == self.one2K)
            grad[0::2] = np.einsum('ik,ikj,i->kj', mask, self.phi, resid).ravel()
            mask = (maxk[1][:, None] == self.one2K)
            grad[1::2] = -np.einsum('ik,ikj,i->kj', mask, self.phi, resid).ravel()
        else:
            mask = (maxk[:, None] == self.one2K)
            grad = np.einsum('ik,ikj,i->kj', mask, self.phi, resid).ravel()
        grad /= n
        # assert len(wmax) == nparams_pwf-1
        if maxwne > 0.0 and self.L_regularizer > 0.0:
            grad = self._jac_maxwne(grad, K, weights, maxwne,
                                    wmax, wnorms, wnormk, self.L_regularizer)
        if wsum is not None:
            d = self.centers.shape[1]
            grad += _calc_cvx_wsum_grad(wsum, self.con_penalty, K, d, self.variant)
        if self.L_sum_regularizer > 0.0:
            grad += _L_sum_reg_grad(weights, self.L_sum_regularizer,
                                    self.is_symmetrized, nparams_pwf)
        return grad


class SmoothMaxMinAffineOptimTask(OptimTask):
    def __init__(
        self, is_symmetrized, X, y, K,
        L_sum_regularizer, L_regularizer, L_regularizer_offset,
        con_penalty=1e6, smooth_tol=1e-6, verbose=False, adj_f=False,
    ):
        super().__init__(verbose)
        self.is_symmetrized = is_symmetrized
        self.X1 = np.insert(X, 0, 1.0, axis=1)
        self.y = y
        self.K = K
        self.L_sum_regularizer = L_sum_regularizer
        self.L_regularizer = L_regularizer
        self.L_regularizer_offset = L_regularizer_offset
        self.con_penalty = con_penalty
        self.smooth_tol = smooth_tol
        self.verbose = verbose
        self.adj_f = adj_f

        n, d = X.shape
        self._range1 = np.arange(n)
        self._range2 = np.arange(K)
        self.cache_key = None

    def _wcache(self, weights):
        n, d1 = self.X1.shape
        mu = self.smooth_tol
        W = weights.reshape((self.K, -1, d1))
        wnorms = np.linalg.norm(W[:, :, 1:], axis=2).ravel()
        XW = W.dot(self.X1.T)
        Ind = np.argmin(XW, axis=1)
        Val = XW[self._range2[:, None], Ind, self._range1]
        ind = np.argmax(Val, axis=0)
        yhat = Val[ind, self._range1]
        if self.adj_f:
            D1, D2, D2sum = _calc_D12sum(XW, Val, yhat, mu)
            yhat += mu * np.log(np.sum(D1 / D2sum, axis=0))
        else:
            D1 = D2 = D2sum = None
        wns, maxwne = _calc_logsumexp_maxwne(
            wnorms, mu, self.L_regularizer_offset)
        return yhat, wnorms, maxwne, D1, D2, D2sum, XW, Val, wns

    def _cache(self, weights):
        self.cache_key = weights
        if self.is_symmetrized:
            wlen = int(0.5 * len(weights))
            self.cache = (
                self._wcache(weights[:wlen]),
                self._wcache(weights[wlen:]),
            )
        else:
            self.cache = self._wcache(weights)

    def fun(self, weights):
        super().fun(weights)
        self._cache(weights)
        if self.is_symmetrized:
            yhat0, wnorms0, maxwne0 = self.cache[0][:3]
            yhat1, wnorms1, maxwne1 = self.cache[1][:3]
            resid = yhat0 - yhat1
            resid -= self.y
            maxwne = max(maxwne0, maxwne1)
            wnorms = np.concatenate([wnorms0, wnorms1])
        else:
            yhat, wnorms, maxwne = self.cache[:3]
            resid = yhat - self.y
        f = 0.5 * (resid.dot(resid)/len(resid) + self.L_regularizer * (maxwne ** 2))
        if self.L_sum_regularizer is not None:
            f += (0.5 * self.L_sum_regularizer) * wnorms.dot(wnorms)
        return f

    def _calc_grad_part1(self, XW, Val, yhat, D1, D2, D2sum):
        if D1 is None:
            D1, D2, D2sum = _calc_D12sum(XW, Val, yhat, self.smooth_tol)
        D2norm = np.divide(D2, D2sum[:, None, :], out=D2)
        D3 = np.divide(D1, D2sum, out=D1)
        D3 /= D3.sum(axis=0)[None, :]
        D3 = np.multiply(D2norm, D3[:, None, :], out=D2norm)
        return D3

    def _calc_grad_part2(self, weights, maxwne, wnorms, wns, D3, resid,
                         is_neg=False, is_maxwne=True):
        grad = np.tensordot(
            self.X1[None, None, :, :] * D3[:, :, :, None],
            resid,
            axes=([2], [0]),
        ).ravel()
        grad /= self.X1.shape[0]
        if is_neg:
            np.negative(grad, out=grad)

        d = self.X1.shape[1] - 1
        if is_maxwne and maxwne > 0.0 and self.L_regularizer > 0.0:
            grad = _calc_logsumexp_maxwne_jac(
                grad, (self.K, 2*d), weights, wnorms, maxwne, wns,
                False, self.L_regularizer)
        if self.L_sum_regularizer > 0.0:
            grad += _L_sum_reg_grad(weights, self.L_sum_regularizer, False, 1+d)
        return grad
        

    def jac(self, weights):
        super().jac(weights)
        if self.cache_key is None or not np.array_equiv(self.cache_key, weights):
            self._cache(weights)
        self.cache_key = None  # cache objects are modified below
        if self.is_symmetrized:
            yhat0, wnorms0, maxwne0, D01, D02, D02sum, XW0, Val0, wns0 = self.cache[0]
            D03 = self._calc_grad_part1(XW0, Val0, yhat0, D01, D02, D02sum)
            yhat1, wnorms1, maxwne1, D11, D12, D12sum, XW1, Val1, wns1 = self.cache[1]
            D13 = self._calc_grad_part1(XW1, Val1, yhat1, D11, D12, D12sum)
            resid = np.subtract(np.subtract(yhat0, yhat1, out=yhat0), self.y, out=yhat0)
            wlen = int(0.5 * len(weights))
            is_maxwne = (maxwne0 >= maxwne1)
            grad0 = self._calc_grad_part2(
                weights[:wlen], maxwne0, wnorms0, wns0, D03, resid,
                is_maxwne=is_maxwne,
            )
            grad1 = self._calc_grad_part2(
                weights[wlen:], maxwne1, wnorms1, wns1, D13, resid,
                is_maxwne=(not is_maxwne), is_neg=True,
            )
            grad = np.concatenate([grad0, grad1])
        else:
            yhat, wnorms, maxwne, D1, D2, D2sum, XW, Val, wns = self.cache
            D3 = self._calc_grad_part1(XW, Val, yhat, D1, D2, D2sum)
            resid = np.subtract(yhat, self.y, out=yhat)
            grad = self._calc_grad_part2(weights, maxwne, wnorms, wns, D3, resid)
        return grad


def _L_sum_reg_grad(weights, L_sum_regularizer, is_symmetrized, nparams_pwf):
    w = L_sum_regularizer * weights
    if is_symmetrized:
        nparams_pwf2 = nparams_pwf * 2
        w[0::nparams_pwf2] = 0.0
        w[1::nparams_pwf2] = 0.0
    else:
        w[::nparams_pwf] = 0.0
    return w


def _calc_cvx_wsum(weights, d, is_convex, variant):
    wsum = None
    if is_convex:
        if variant == '+': 
            d1 = d+1
            wsum = weights[:, 1:d1] + weights[:, d1:]
        else:
            wsum = weights[:, -1:]
        np.minimum(0.0, wsum, out=wsum)
        if np.min(wsum) >= 0.0:
            wsum = None
    return wsum


def _calc_cvx_wsum_grad(wsum, con_penalty, K, d, variant):
    wsum *= -con_penalty
    if variant == '+':
        wsum_grad = np.hstack([np.zeros((K, 1)), wsum, wsum])
    else:
        wsum_grad = np.hstack([np.zeros((K, 1+d)), wsum])
    return wsum_grad.ravel()


def _calc_D12sum(XW, Val, yhat, mu):
    """Helper for smooth MMA model calculations."""
    # Calculating: D1 = np.exp((Val - yhat[None, :]) / mu)
    D1 = Val - yhat[None, :]
    D1 /= mu
    np.exp(D1, out=D1)    
    # Calculating: D2 = np.exp((Val[:, None, :] - XW) / mu)
    D2 = Val[:, None, :] - XW
    D2 /= mu
    np.exp(D2, out=D2)

    D2sum = np.sum(D2, axis=1)
    return D1, D2, D2sum


def _calc_logsumexp_maxwne(wnorms, mu, L_regularizer_offset):
    """
    Approximate max(0, max(wnorms) - L_regularizer_offset) by
    applying log(sum(exp(wnorms))) approximation of the inner max funxtion.
    """
    maxwnorm = np.max(wnorms)
    wns = wnorms - maxwnorm
    wns /= mu
    np.exp(wns, out=wns)
    logsumexp = maxwnorm + mu * np.log(np.sum(wns))
    wns /= np.sum(wns)
    maxwne = max(0.0, logsumexp - L_regularizer_offset)
    return wns, maxwne


def _calc_maxwne_softmax(K, weights, wnorms, maxwne, wns, L_regularizer):
    """
    Calculates the gradient of _calc_logsumexp_maxwne(...) ** 2 for a single-sided
    DCF or MMA model (recall that softmax is the derivative of log-sum-exp).
    """
    if isinstance(K, tuple):
        g = weights.reshape(list(K) + [-1], copy=True)
        g[:, :, 0] = 0.0
        g = g.reshape((-1, g.shape[-1]))
    else:
        g = weights.reshape((K, -1), copy=True)
        g[:, 0] = 0.0
    g /= wnorms[:, None]
    g *= (maxwne * L_regularizer)
    g *= wns[:, None]
    return g.ravel()


def _calc_logsumexp_maxwne_jac(grad, K, weights, wnorms, maxwne, wns,
                               is_symmetrized, L_regularizer):
    if is_symmetrized:
        grad[0::2] += _calc_maxwne_softmax(
            K, weights[0::2], wnorms[:K], maxwne, wns[:K], L_regularizer)
        grad[1::2] += _calc_maxwne_softmax(
            K, weights[1::2], wnorms[K:], maxwne, wns[K:], L_regularizer)
    else:
        grad += _calc_maxwne_softmax(K, weights, wnorms, maxwne, wns, L_regularizer)
    return grad


def _test_smooth_optim_task():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> from ai.gandg.common.partition import rand_voronoi_partition, find_min_dist_centers
    >>> from ai.gandg.algorithm.dcf.dcf import _dcf_calc_phi

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])
    >>> n, d = X.shape

    >>> K = 5
    >>> weights1 = np.random.randn(K, d+2).ravel()
    >>> P = rand_voronoi_partition(K, X)
    >>> cidx = find_min_dist_centers(X, P)
    >>> len(cidx) == K
    True
    >>> centers = X[cidx, :]
    
    >>> variant = '2'
    >>> sot1 = SmoothDCFLocalOptimTask(
    ...     centers=centers,
    ...     phi=_dcf_calc_phi(X, centers, variant, True),
    ...     y=y,
    ...     variant=variant,
    ...     is_convex=False,
    ...     is_symmetrized=False,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p1 = sot1.fun(weights1)
    >>> np.round(p1, decimals=4)
    8.5123
    >>> g1 = sot1.jac(weights1)
    >>> np.round(g1, decimals=6)
    array([ 0.000000e+00,  3.679000e-03,  4.700000e-05, -2.132400e-02,
            1.277010e+00,  3.355553e+00,  8.872500e-01,  3.517649e+00,
            2.554640e-01,  3.180020e-01, -6.921400e-01,  7.323230e-01,
            6.194940e-01, -1.770960e+00, -3.175200e-01,  1.807908e+00,
            1.037826e+00,  1.260986e+00,  1.559265e+00,  2.098880e+00])
    >>> np.round(sot1.jac_finite_difference(weights1) - g1 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0.])

    >>> variant = '+'
    >>> weights2 = np.random.randn(K, 1+2*d).ravel()
    >>> sot2 = SmoothDCFLocalOptimTask(
    ...     centers=centers,
    ...     phi=_dcf_calc_phi(X, centers, variant, True),
    ...     y=y,
    ...     variant=variant,
    ...     is_convex=False,
    ...     is_symmetrized=False,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p2 = sot2.fun(weights2)
    >>> np.round(p2, decimals=4)
    3.0544
    >>> g2 = sot2.jac(weights2)
    >>> np.round(g2, decimals=6)
    array([ 0.487715,  0.71809 ,  1.213847, -0.228823, -0.202124,  0.      ,
            0.007226, -0.017663,  0.021096, -0.009145,  0.      ,  0.015854,
            0.003508,  0.009679, -0.023577, -0.212086,  0.04581 , -0.062759,
           -0.371522, -0.628068,  0.146561,  0.384821,  0.259903,  0.03094 ,
            0.003662])
    >>> np.round(sot2.jac_finite_difference(weights2) - g2 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])

    >>> variant = '+'
    >>> weights3 = np.random.randn(K, 2*(1+2*d)).ravel()
    >>> sot3 = SmoothDCFLocalOptimTask(
    ...     centers=centers,
    ...     phi=_dcf_calc_phi(X, centers, variant, True),
    ...     y=y,
    ...     variant=variant,
    ...     is_convex=False,
    ...     is_symmetrized=True,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p3 = sot3.fun(weights3)
    >>> np.round(p3, decimals=4)
    14.0246
    >>> g3 = sot3.jac(weights3)
    >>> np.round(g3, decimals=6)
    array([ 0.000000e+00, -1.845080e-01, -2.484500e-02, -2.871490e-01,
            9.729000e-03, -3.072710e-01,  3.198000e-03, -1.096700e-02,
           -1.939700e-02,  8.579000e-03, -7.886100e-01,  5.703080e-01,
           -9.369540e-01,  8.656040e-01, -4.966850e-01, -7.552300e-02,
           -8.533000e-02, -4.761700e-02, -7.759300e-02,  1.775448e+00,
            5.268320e-01,  2.460910e+00,  9.123490e-01, -4.812800e-02,
            1.789000e-03,  4.245700e-02, -9.100000e-03,  2.589002e+00,
            3.814840e-01,  3.918733e+00,  0.000000e+00, -3.842800e-02,
           -8.235000e-03,  9.336000e-03,  6.905000e-03, -9.103000e-03,
           -1.909200e-02, -1.127200e-02,  1.699000e-02, -3.015300e-02,
           -2.546503e+00,  0.000000e+00, -3.583830e-01, -2.278300e-02,
           -1.893650e+00,  1.823000e-03, -2.042164e+00,  1.832900e-02,
           -5.232470e-01, -1.181000e-02])
    >>> np.round(sot3.jac_finite_difference(weights3) - g3 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    pass


def _test_nonsmooth_optim_task():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> from ai.gandg.common.partition import rand_voronoi_partition, find_min_dist_centers
    >>> from ai.gandg.algorithm.dcf.dcf import _dcf_calc_phi

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])
    >>> n, d = X.shape

    >>> K = 5
    >>> weights = np.random.randn(K, X.shape[1]).ravel()

    >>> P = rand_voronoi_partition(K, X)
    >>> cidx = find_min_dist_centers(X, P)
    >>> centers = X[cidx, :]
    
    >>> variant = '2'
    >>> got1 = NonSmoothDCFLocalOptimTask(
    ...     centers=centers,
    ...     phi=_dcf_calc_phi(X, centers, variant, False),
    ...     y=y,
    ...     variant=variant,
    ...     is_convex=False,
    ...     is_symmetrized=False,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p1 = got1.fun(weights)
    >>> np.round(p1, decimals=4)
    6.7449
    >>> g1 = got1.jac(weights)
    >>> np.round(g1, decimals=6)
    array([ 0.000000e+00,  3.679000e-03,  0.000000e+00, -2.132400e-02,
            9.895940e-01,  3.454384e+00,  0.000000e+00,  1.020900e-02,
            2.072485e+00,  5.364934e+00])
    >>> np.round(got1.jac_finite_difference(weights) - g1 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    >>> variant = '+'
    >>> weights2 = np.random.randn(K, 1+2*d).ravel()
    >>> got2 = NonSmoothDCFLocalOptimTask(
    ...     centers=centers,
    ...     phi=_dcf_calc_phi(X, centers, variant, True),
    ...     y=y,
    ...     variant=variant,
    ...     is_convex=False,
    ...     is_symmetrized=False,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p2 = got2.fun(weights2)
    >>> np.round(p2, decimals=4)
    3.9703
    >>> g2 = got2.jac(weights2)
    >>> np.round(g2, decimals=6)
    array([ 0.      ,  0.009917, -0.003147,  0.009234, -0.003819,  0.      ,
           -0.024148, -0.01457 , -0.020874,  0.009697, -0.422496, -0.459981,
           -0.007819, -0.018364, -0.523877,  0.221276,  0.638287, -0.010905,
           -0.037867,  0.356857,  0.795751,  0.004662,  1.173281,  0.012949,
            0.003784])
    >>> np.round(got2.jac_finite_difference(weights2) - g2 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])

    >>> variant = '+'
    >>> weights3 = np.random.randn(K, 2*(1+2*d)).ravel()
    >>> got3 = NonSmoothDCFLocalOptimTask(
    ...     centers=centers,
    ...     phi=_dcf_calc_phi(X, centers, variant, True),
    ...     y=y,
    ...     variant=variant,
    ...     is_convex=False,
    ...     is_symmetrized=True,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p3 = got3.fun(weights3)
    >>> np.round(p3, decimals=4)
    12.5344
    >>> g3 = got3.jac(weights3)
    >>> np.round(g3, decimals=6)
    array([-9.502200e-02,  0.000000e+00, -4.851200e-02, -6.228000e-03,
           -3.911100e-02,  1.626300e-02, -2.386400e-02, -7.358000e-03,
           -3.264800e-02, -6.086000e-03, -3.013548e+00,  0.000000e+00,
           -4.858500e-02, -1.797400e-02, -1.862617e+00,  2.501800e-02,
           -2.576146e+00, -2.454600e-02, -1.147347e+00, -9.064000e-03,
           -3.343150e-01,  3.186278e+00, -3.581100e-02,  5.581676e+00,
           -7.425300e-02,  3.245275e+00, -1.277070e-01, -3.823600e-02,
           -1.784560e-01,  1.652640e-01,  0.000000e+00,  0.000000e+00,
            8.320000e-04,  3.041100e-02, -5.294000e-03,  4.148000e-03,
           -6.763000e-03, -6.169000e-03, -3.163800e-02, -8.935000e-03,
           -8.379650e-01,  1.094570e+00,  2.408200e-02, -7.460000e-03,
           -1.632206e+00,  2.475710e-01, -8.162810e-01,  2.848757e+00,
           -9.956800e-02,  7.361740e-01])
    >>> np.round(got3.jac_finite_difference(weights3) - g3 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    pass


def test_mma_optim_task():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> from ai.gandg.common.partition import rand_voronoi_partition, find_min_dist_centers

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])
    >>> n, d = X.shape

    >>> K = 5
    >>> d = X.shape[1]
    >>> W = np.random.randn(K, d*2, 1+d)
    >>> yhat = (W[:, :, 1:].dot(X.T) + W[:, :, 0][:, :, None]).min(axis=1).max(axis=0)
    >>> resid = y - yhat
    >>> np.round(0.5 * resid.dot(resid) / len(resid), decimals=4)
    6.4306
    >>> weights = W.ravel()
    >>> mot1 = SmoothMaxMinAffineOptimTask(
    ...     is_symmetrized=False, X=X, y=y, K=K,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p1 = mot1.fun(weights)
    >>> np.round(p1, decimals=4)
    6.9105
    >>> np.round(
    ...     SmoothMaxMinAffineOptimTask(
    ...         is_symmetrized=False, X=X, y=y, K=K,
    ...         L_regularizer=0.01,
    ...         L_regularizer_offset=0.1,
    ...         L_sum_regularizer=0.02,
    ...         adj_f=True,
    ...     ).fun(weights),
    ...     decimals=4,
    ... )
    6.9105
    >>> g1 = mot1.jac(weights)
    >>> np.round(g1, decimals=4)
    array([-2.030e-01,  7.400e-02, -4.097e-01, -2.295e-01, -1.681e-01,
            3.438e-01,  0.000e+00,  1.020e-02,  4.900e-03,  1.073e-01,
            2.638e-01, -1.170e-02, -2.024e-01, -8.760e-02,  4.072e-01,
           -1.637e-01,  8.810e-02,  4.333e-01,  0.000e+00,  3.090e-02,
           -3.200e-02,  0.000e+00,  1.100e-02, -2.380e-02,  0.000e+00,
            5.900e-03, -6.000e-04, -7.500e-03,  7.300e-03, -2.700e-02,
           -2.422e-01,  2.590e-02,  2.457e-01, -1.801e-01,  8.090e-02,
            8.410e-02, -1.332e-01,  1.018e-01,  2.004e-01, -1.313e-01,
            1.850e-01, -1.437e-01, -3.250e-02, -3.420e-02,  4.830e-02,
            0.000e+00,  7.000e-04,  3.290e-02,  0.000e+00, -2.360e-02,
            3.190e-02,  0.000e+00, -3.000e-02,  8.300e-03, -4.602e-01,
            6.483e-01, -1.368e-01, -3.865e-01,  4.504e-01,  4.863e-01])
    >>> np.round(mot1.jac_finite_difference(weights) - g1 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0.])

    >>> W2 = np.random.randn(K, d*2, 1+d)
    >>> weights2 = np.concatenate([W.ravel(), W2.ravel()])
    >>> mot2 = SmoothMaxMinAffineOptimTask(
    ...     is_symmetrized=True, X=X, y=y, K=K,
    ...     L_regularizer=0.01,
    ...     L_regularizer_offset=0.1,
    ...     L_sum_regularizer=0.02,
    ... )
    >>> p2 = mot2.fun(weights2)
    >>> np.round(p2, decimals=4)
    5.5646
    >>> g2 = mot2.jac(weights2)
    >>> np.round(g2, decimals=4)
    array([-7.160e-02,  1.289e-01, -2.229e-01, -1.180e-01, -8.840e-02,
            2.582e-01,  0.000e+00,  1.020e-02,  4.900e-03,  8.450e-02,
            2.019e-01, -3.900e-03, -1.522e-01, -7.540e-02,  3.138e-01,
           -1.433e-01,  8.600e-02,  3.831e-01,  0.000e+00,  3.090e-02,
           -3.200e-02,  0.000e+00,  1.100e-02, -2.380e-02,  0.000e+00,
            5.900e-03, -6.000e-04,  9.900e-03,  1.230e-02, -2.190e-02,
           -1.974e-01,  1.750e-02,  2.065e-01, -1.351e-01,  7.140e-02,
            7.340e-02, -1.223e-01,  9.270e-02,  1.866e-01, -1.147e-01,
            1.606e-01, -1.260e-01, -2.050e-02, -3.250e-02,  3.600e-02,
            0.000e+00,  7.000e-04,  3.290e-02,  0.000e+00, -2.360e-02,
            3.190e-02,  0.000e+00, -3.000e-02,  8.300e-03, -3.805e-01,
            6.001e-01, -8.490e-02, -3.460e-01,  4.379e-01,  4.871e-01,
            4.400e-02, -8.730e-02,  4.510e-02,  0.000e+00, -2.220e-02,
           -1.350e-02,  2.420e-02, -6.170e-02,  3.600e-03,  0.000e+00,
            1.060e-02,  2.850e-02, -1.800e-02, -2.850e-02, -2.800e-02,
            2.940e-02,  3.040e-02, -9.080e-02,  0.000e+00, -7.100e-03,
           -3.120e-02,  0.000e+00,  2.160e-02, -1.340e-02,  3.045e-01,
           -3.308e-01, -1.298e-01,  2.669e-01, -4.619e-01,  6.370e-02,
            3.798e-01, -8.440e-02, -4.873e-01,  4.112e-01, -3.936e-01,
           -8.456e-01,  0.000e+00,  1.000e-03,  7.600e-03,  0.000e+00,
           -1.200e-03,  2.690e-02,  1.340e-02, -3.700e-03,  5.700e-03,
            2.031e-01, -8.560e-02,  3.534e-01, -9.430e-02, -1.264e-01,
           -1.098e-01,  0.000e+00,  1.370e-02,  3.150e-02,  1.959e-01,
            8.590e-02, -1.451e-01, -5.300e-02, -1.281e-01, -2.520e-02])
    >>> np.round(mot2.jac_finite_difference(weights2) - g2 + 1e-7, decimals=6)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0.])
    """
    pass

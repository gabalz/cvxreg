import numpy as np

from cvxreg.optim.optim_task import OptimTask


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
        from cvxreg.algorithm.dcf.dcf import _dcf_calc_Xw
        v = _dcf_calc_Xw(self.phi, weights)
        yhat = np.max(v, axis=1)
        v -= yhat[:, None]
        v /= self.smooth_tol
        np.exp(v, out=v)
        v /= np.sum(v, axis=1)[:, None]
        return v, yhat

    def _cache(self, weights):
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
        from cvxreg.algorithm.dcf.dcf import _dcf_calc_Xw
        Xw = _dcf_calc_Xw(self.phi, weights)
        maxk = np.argmax(Xw if noise is None else Xw+noise[None, :], axis=1)
        yhat = Xw[np.arange(Xw.shape[0]), maxk]
        return maxk, yhat

    def _cache(self, weights):
        noise = None
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

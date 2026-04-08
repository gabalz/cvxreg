import numpy as np

from ai.gandg.cvxreg.common.distance import euclidean_distance, squared_distance
from ai.gandg.cvxreg.common.partition import Partition, voronoi_partition


FPC_FIRST_IDX__ZERO = 'zero'
FPC_FIRST_IDX__RANDOM = 'random'
FPC_FIRST_IDX__CLOSEST_TO_CENTER = 'closest_to_center'
FPC_FIRST_IDX__DEFAULT = FPC_FIRST_IDX__CLOSEST_TO_CENTER


def _fpc_first_center_idx(data, first_idx, dist):
    if first_idx == FPC_FIRST_IDX__CLOSEST_TO_CENTER:
        i = np.argmin(dist(data, np.mean(data, axis=0)))
    elif first_idx == FPC_FIRST_IDX__ZERO:
        i = 0
    elif first_idx == FPC_FIRST_IDX__RANDOM:
        i = np.random.randint(data.shape[0])
    else:
        raise NotImplementedError('Not supported first_idx: {}'.format(first_idx))

    return i


def farthest_point_clustering(
    ncells, data, dist=squared_distance, first_idx=FPC_FIRST_IDX__DEFAULT,
):
    """Computes a partition by the farthest-point clustering (FPC) algorithm.

    :param ncells: number of cells to find
    :param data: data matrix (each row is a sample)
    :param dist: distance function
    :param first_idx: decides how the first index is selected
    :return: partition represented by the sample indices (Partition object)
    """
    n, d = data.shape
    center_idxs = [_fpc_first_center_idx(data, first_idx, dist)]

    min_dists = dist(data, data[center_idxs[0], :])
    for k in range(1, ncells):
        if len(center_idxs) == n:
            break

        maxi = np.argmax(min_dists)
        center_idxs.append(maxi)

        maxi_dists = dist(data, data[maxi, :])
        min_dists = np.minimum(min_dists, maxi_dists)

    return voronoi_partition(data[center_idxs, :], data)


def get_data_radius(data, dist=euclidean_distance):
    if dist == euclidean_distance:
        _dist = squared_distance
    else:
        _dist = dist
    r = max(_dist(data, np.mean(data, axis=0)))
    if dist == euclidean_distance:
        r = np.sqrt(r)
    return r


def adaptive_farthest_point_clustering(
    data, dist=euclidean_distance, q=1,
    first_idx=FPC_FIRST_IDX__DEFAULT,
    return_center_idxs=False,
):
    """Computes a partition of the adaptive farthest-point clustering (AFPC) algorithm.

    :param data: data matrix (each row is a sample)
    :param dist: distance function
    :param q: scaler for stopping condition
    :param first_idx: decides how the first index is selected
    :param return_center_idxs: if True, center_idxs also returned
    :return: partition represented by the sample indices (Partition object)
    """
    n, d = data.shape
    data_radius = get_data_radius(data, dist)
    if data_radius < 1e-16:
        partition = Partition(npoints=n, ncells=1, cells=(np.arange(n),))
        if return_center_idxs:
            return partition, [0]
        return partition
    center_idxs = [_fpc_first_center_idx(data, first_idx, dist)]

    q2 = 2*q
    maxK = int(np.ceil(n**(d/(q2+d))))
    min_center_dists = dist(data, data[center_idxs[0], :])
    closest_centers = np.zeros(n, dtype=int)
    update_cond = np.empty(n, dtype=bool)
    for K in range(1, 1+maxK):
        if len(center_idxs) == n:
            break

        radius = np.max(min_center_dists)
        if n * (radius/data_radius)**q2 <= K:
            break

        maxi = np.argmax(min_center_dists)
        center_idxs.append(maxi)

        maxi_dists = dist(data, data[maxi, :])
        np.greater(min_center_dists, maxi_dists, out=update_cond)
        np.putmask(closest_centers, update_cond, K)
        np.minimum(min_center_dists, maxi_dists, out=min_center_dists)

    cells = tuple([np.where(closest_centers == k)[0] for k in range(len(center_idxs))])
    partition = Partition(npoints=n, ncells=len(cells), cells=cells)
    if return_center_idxs:
        return partition, center_idxs
    return partition

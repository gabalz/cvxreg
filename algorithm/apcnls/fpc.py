import numpy as np

from common.distance import euclidean_distance, squared_distance
from common.partition import Partition, voronoi_partition, cell_radiuses


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

    >>> X = np.array(
    ...     [[1., 1.], [-1., 1.], [0., 1.],
    ...      [-1.5, 0.5], [0.5, -1.], [1., 1.5],
    ...      [-1., -1.], [-2., -1], [0., 1.]],
    ... )
    >>> p = farthest_point_clustering(ncells=3, data=X)
    >>> p.npoints
    9
    >>> p.ncells
    3
    >>> p.cells
    ([0, 1, 2, 3, 5, 8], [6, 7], [4])

    >>> from common.util import set_random_seed
    >>> set_random_seed(19)
    >>> C = np.random.randn(5, 3)
    >>> C
    array([[-1.03525403,  0.41239449,  0.70404776],
           [-0.12290513,  0.12498242, -1.43440465],
           [-0.05608202,  0.20520597, -1.44680874],
           [ 0.49377073,  0.22724934, -0.97081668],
           [-0.17539455, -1.07307641, -0.72220697]])
    >>> X = C[np.random.randint(C.shape[0], size=(100,)), :]
    >>> X += np.random.randn(*X.shape) * 0.1
    >>> p = farthest_point_clustering(ncells=5, data=X)
    >>> p.npoints
    100
    >>> p.ncells
    5
    >>> [len(c) for c in p.cells]
    [31, 15, 22, 22, 10]
    >>> np.vstack([np.mean(X[c, :], axis=0).T for c in p.cells])
    array([[-0.07134101,  0.16107547, -1.42839121],
           [-1.04114572,  0.41896789,  0.70097091],
           [-0.17329119, -1.03270566, -0.71894531],
           [ 0.47347632,  0.22806483, -0.96611127],
           [-0.0059361 ,  0.15846984, -1.59665994]])
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

    >>> from common.util import set_random_seed
    >>> set_random_seed(19)
    >>> X = np.random.randn(200, 2)
    >>> partition = adaptive_farthest_point_clustering(data=X)
    >>> partition.npoints
    200
    >>> partition.ncells
    15
    >>> [len(c) for c in partition.cells]
    [63, 1, 5, 3, 1, 5, 4, 3, 13, 17, 10, 14, 16, 23, 22]
    >>> [int(np.mean(c)) for c in partition.cells]
    [107, 117, 110, 166, 71, 110, 137, 81, 107, 111, 86, 67, 83, 100, 81]
    >>> np.round(max(cell_radiuses(X, partition)), decimals=4)
    0.8883

    >>> C = np.random.randn(5, 3)
    >>> C
    array([[-0.57681049, -1.0982188 , -1.77021275],
           [-0.29580036,  0.55481809, -1.40079008],
           [-0.47390752, -0.02820206, -0.79310281],
           [-0.5338925 ,  0.07452468,  0.08311392],
           [-0.84831213, -1.02133226,  0.38650518]])
    >>> X = C[np.random.randint(C.shape[0], size=(100,)), :]
    >>> X += np.random.randn(*X.shape) * 0.1
    >>> partition = adaptive_farthest_point_clustering(data=X, q=2)
    >>> partition.npoints
    100
    >>> partition.ncells
    5
    >>> [len(c) for c in partition.cells]
    [15, 17, 22, 21, 25]
    >>> np.vstack([np.mean(X[c, :], axis=0).T for c in partition.cells])
    array([[-0.49366699, -0.04686255, -0.77307882],
           [-0.61269755, -1.07736845, -1.77314976],
           [-0.86476045, -0.98983781,  0.40252153],
           [-0.28370596,  0.52195337, -1.41375348],
           [-0.54848627,  0.05573469,  0.06377456]])
    """
    n, d = data.shape
    data_radius = get_data_radius(data, dist)
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

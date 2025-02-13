import numpy as np

from scipy.spatial import distance
from common.distance import euclidean_distance, squared_distance


def find_min_dist_centers(data, partition, dist=squared_distance, op=np.max):
    """Find the centers minimizing the distance for a partition.

    :param data: data matrix (each row is a sample)
    :param partition: partition represented by sample indices (Partition object)
    :param dist: distance function
    :return: tuple of center indices

    >>> X = np.array(
    ...     [[1., 1.], [-1., 1.], [0., 1.],
    ...      [-1.5, 0.5], [0.5, -1.], [1., 1.5],
    ...      [-1., -1.], [-2., -1], [0., 1.]],
    ... )
    >>> p = Partition(npoints=9, ncells=3, cells=([0, 1, 2, 3, 5, 8], [6, 7], [4]))
    >>> centers = find_min_dist_centers(X, p)
    >>> centers
    (2, 6, 4)
    >>> for k in range(p.ncells):
    ...     cell = p.cells[k]
    ...     print(squared_distance(X[cell, :], X[centers[k], :]))
    [1.   1.   0.   2.5  1.25 0.  ]
    [0. 1.]
    [0.]
    """
    p = 1
    if dist == squared_distance:
        dist = 'sqeuclidean'
        if op != np.max:
            p = 2

    center_idxs = []
    for k in range(partition.ncells):
        cell = partition.cells[k]
        dists = distance.cdist(data[cell, :], data[cell, :], dist)
        center_idxs.append(cell[np.argmin(op(dists**p, axis=1))])
    return tuple(center_idxs)


def cell_radiuses(data, partition, centers=None, dist=euclidean_distance):
    """Calculate all cell radiuses within a partition.

    :param data: data matrix (each row is a sample)
    :param partition: partition represented by sample indices (Partition object)
    :param centers: centers (optional, computed if not provided)
    :param dist: distance function
    :return: cell radiuses

    >>> X = np.array(
    ...     [[1., 1.], [-1., 1.], [0., 1.],
    ...      [-1.5, 0.5], [0.5, -1.], [1., 1.5],
    ...      [-1., -1.], [-2., -1], [0., 1.]],
    ... )
    >>> p = Partition(npoints=9, ncells=3, cells=([0, 1, 2, 3, 5, 8], [6, 7], [4]))
    >>> cell_radiuses(X, p, dist=squared_distance)
    (2.5, 1.0, 0.0)
    >>> tuple(np.round(cell_radiuses(X, p), decimals=4))
    (1.5811, 1.0, 0.0)
    """
    if dist == euclidean_distance:
        _dist = squared_distance
    else:
        _dist = dist
    if centers is None:
        centers = data[find_min_dist_centers(data, partition, _dist), :]
    assert partition.ncells == centers.shape[0]
    r = tuple([
        max(_dist(data[cell, :], centers[center_idx, :]))
        for cell, center_idx in zip(partition.cells, range(centers.shape[0]))
    ])
    if dist == euclidean_distance:
        r = tuple([np.sqrt(v) for v in r])
    return r


def voronoi_partition(centers, data, dist=squared_distance):
    """Returns a Voronoi partition of the data around the specified centers with respect to the Euclidean distance.
    points X around centers C based on the Euclidean distance.

    :param centers: center matrix (each row is a center point)
    :param data: data matrix (each row is a sample)
    :param dist: distance function
    :return: Partition object representing the Voronoi partition around the given centers

    >>> centers = np.array([[1., 0.], [-1., 0.]])
    >>> data = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1]])
    >>> partition = voronoi_partition(centers, data)
    >>> partition.npoints
    5
    >>> partition.ncells
    2
    >>> sorted(partition.cells)
    [[0, 2], [1, 3, 4]]
    """
    npoints = data.shape[0]
    ncells = centers.shape[0]
    cell_idx = np.zeros(npoints, dtype=int)
    cell_center_dist = dist(data, centers[0, :])
    for k in range(1, ncells):
        center = centers[k, :]
        center_dist = dist(data, center)
        cell_idx = np.where(cell_center_dist <= center_dist, cell_idx, k)
        np.minimum(cell_center_dist, center_dist, out=cell_center_dist)
    cells = [np.where(cell_idx == k)[0] for k in range(ncells)]
    cells = tuple([list(cell) for cell in cells if len(cell) > 0])
    return Partition(npoints=npoints, ncells=len(cells), cells=cells)


def rand_voronoi_partition(ncenters, data, dist=squared_distance):
    """Returns a Voronoi partition of the data around uniformly drawn centers.

    :param ncenters: number of distinct centers to be drawn uniformly
    :param data: data matrix (each row is a sample)
    :param dist: distance function
    :return: Partition object representing the randomly created Voronoi partition

    >>> from common.util import set_random_seed
    >>> set_random_seed(19)
    
    >>> data = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1], [0., 1.]])
    >>> partition = rand_voronoi_partition(2, data)
    >>> partition.npoints
    6
    >>> partition.ncells
    2
    >>> sorted(partition.cells)
    [[0, 1, 2, 5], [3, 4]]
    """
    indices = np.random.permutation(data.shape[0])[:ncenters]
    centers = data[indices, :]
    return voronoi_partition(centers, data, dist)


def max_affine_partition(data, maf):
    """Returns the induced partition by a max-affine function.

    :param data: data matrix (each row is a sample)
    :param maf: max-affine function as a matrix (each row is an affine map [offset, slope])
    :returns: Partition object representing the induced partition

    >>> data = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1], [0., 1.]])
    >>> maf = np.array([[1., 0.], [0., -1.], [-1., 1.]])
    >>> p = max_affine_partition(data, maf)
    >>> p.npoints
    6
    >>> p.ncells
    3
    >>> p.cells
    (array([0]), array([3, 4]), array([1, 2, 5]))
    >>> p.assert_consistency()
    >>> p.cell_sizes()
    (1, 2, 3)
    >>> p.cell_indices()
    array([0, 2, 2, 1, 1, 2])

    >>> p == max_affine_partition(data, maf)
    True
    >>> p == singleton_partition(data.shape[0])
    False
    >>> p == Partition(npoints=6, ncells=3, cells=((3, 4), (1, 2, 5), (0,)))
    True
    """
    nhyperplanes = maf.shape[0]
    idx = np.argmax(data.dot(maf.T), axis=1)
    cells = []
    for k in range(nhyperplanes):
        cells.append(np.where(idx == k)[0])
    cells = [c for c in cells if len(c) > 0]
    return Partition(npoints=data.shape[0], ncells=len(cells), cells=tuple(cells))


def singleton_partition(npoints):
    """Returns a singleton partition with cells having exactly one element.

    :param npoints: size of the partitioned set {0, 1, ..., npoints-1}
    :return: Partition object representing the singleton partition ([0], [1], ..., [npoints-1])

    >>> p = singleton_partition(5)
    >>> p.npoints
    5
    >>> p.ncells
    5
    >>> p.cells
    ([0], [1], [2], [3], [4])
    >>> p.assert_consistency()
    >>> p.cell_sizes()
    (1, 1, 1, 1, 1)
    >>> p.cell_indices()
    array([0, 1, 2, 3, 4])
    """
    assert npoints >= 1
    return Partition(npoints=npoints, ncells=npoints, cells=tuple([[i] for i in range(npoints)]))


class Partition(object):
    """Class representation of partitions."""
    __slots__ = ['npoints', 'ncells', 'cells', 'extra']

    def __init__(self, npoints, ncells, cells):
        """Creates a partition over the set {0,...,npoints-1}.
        The partition does not have to be consistent at the stage yet.

        :param npoints: the size of the partitioned set
        :param ncells: the number of cells
        :param cells: the cells
        :return: Partition object with the provided attributes
        """
        self.npoints = npoints
        self.ncells = ncells
        assert ncells <= npoints
        self.cells = cells
        self.extra = {}

    def cell_sizes(self):
        """Returns the number of elements in each cell.
        
        :return: integer tuple representing the cell sizes
        """
        sizes = []
        for cell in self.cells:
            sizes.append(len(cell))
        return tuple(sizes)

    def cell_indices(self):
        """Returns the cell index for each element of the partitioned set.
        
        :return: integer array representing the cell index of each element
        """
        idx = np.empty(self.npoints, dtype=int)
        for i, cell in enumerate(self.cells):
            idx[cell] = i
        return idx

    def assert_consistency(self):
        """Raises an assertion error if the partition is not consistent."""
        assert self.ncells == len(self.cells)
        elems = []
        for k in range(self.ncells):
            cell_k = self.cells[k]
            assert 0 < len(cell_k)
            elems += list(cell_k)
        assert self.npoints == len(elems)
        assert list(range(self.npoints)) == sorted(elems)

    def __eq__(self, other):
        """Test whether two partition are equal (assumes assert_consistency() passes for both partitions)."""
        if not isinstance(other, Partition):
            return NotImplementedError('Cannot compare Partition to {}!'.format(type(other)))

        if self.npoints != other.npoints or self.ncells != other.ncells:
            return False
        for cell, other_cell in zip(sorted([tuple(c) for c in self.cells]),
                                    sorted([tuple(c) for c in other.cells])):
            if len(cell) != len(other_cell):
                return False
            if tuple(cell) != tuple(other_cell):
                return False
        return True

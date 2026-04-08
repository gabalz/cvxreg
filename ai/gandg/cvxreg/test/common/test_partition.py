import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.common.distance import euclidean_distance, squared_distance
from ai.gandg.cvxreg.common.partition import (
    Partition, find_closest_centers, find_min_dist_centers,
    cell_radii, voronoi_partition, rand_voronoi_partition,
    max_affine_partition, singleton_partition,
)

_DATA_X1 = np.array(
    [[1., 1.], [-1., 1.], [0., 1.],
     [-1.5, 0.5], [0.5, -1.], [1., 1.5],
     [-1., -1.], [-2., -1], [0., 1.]],
)
_DATA_X2 = np.array(
    [[1., 15.], [-5., 1.], [0., 30.],
     [-15, -1.5], [0.5, -1.], [-12., -1.5],
     [-1., -1.], [-1., -1], [0., 1.]],
)
_P_9_3 = Partition(npoints=9, ncells=3, cells=([0, 1, 2, 3, 5, 8], [6, 7], [4]))


class TestFindClosestCenters(unittest.TestCase):
    def test_closest_centers(self):
        centers = np.array([[0., 1.], [-2., -1]])
        self.assertEqual(
            find_closest_centers(_DATA_X1, centers),
            (0, 0, 0, 0, 0, 0, 1, 1, 0),
        )


class TestFindMinDistCenters(unittest.TestCase):
    def test_max_op(self):
        centers = find_min_dist_centers(_DATA_X1, _P_9_3)
        self.assertEqual(centers, (2, 6, 4))

    def test_per_cell_squared_distances(self):
        centers = find_min_dist_centers(_DATA_X1, _P_9_3)
        for k, (cell, c) in enumerate(zip(_P_9_3.cells, centers)):
            dists = squared_distance(_DATA_X1[cell, :], _DATA_X1[c, :])
            self.assertEqual(dists.shape, (len(cell),))

    def test_mean_op(self):
        self.assertEqual(find_min_dist_centers(_DATA_X2, _P_9_3, op=np.mean), (1, 6, 4))

    def test_max_op(self):
        self.assertEqual(find_min_dist_centers(_DATA_X2, _P_9_3, op=np.max), (0, 6, 4))


class TestCellRadii(unittest.TestCase):
    def test_squared_distance(self):
        self.assertEqual(cell_radii(_DATA_X1, _P_9_3, dist=squared_distance), (2.5, 1.0, 0.0))

    def test_euclidean_distance(self):
        np.testing.assert_array_almost_equal(
            np.round(cell_radii(_DATA_X1, _P_9_3), decimals=4),
            np.array([1.5811, 1.0, 0.0]),
        )


class TestVoronoiPartition(unittest.TestCase):
    def test_basic(self):
        centers = np.array([[1., 0.], [-1., 0.]])
        data = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1]])
        partition = voronoi_partition(centers, data)
        self.assertEqual(partition.npoints, 5)
        self.assertEqual(partition.ncells, 2)
        self.assertEqual(sorted(partition.cells), [[0, 2], [1, 3, 4]])


class TestRandVoronoiPartition(unittest.TestCase):
    def test_basic(self):
        set_random_seed(19)
        data = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1], [0., 1.]])
        partition = rand_voronoi_partition(2, data)
        self.assertEqual(partition.npoints, 6)
        self.assertEqual(partition.ncells, 2)
        self.assertEqual(sorted(partition.cells), [[0, 1, 2, 5], [3, 4]])


class TestMaxAffinePartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1], [0., 1.]])
        cls.maf = np.array([[1., 0.], [0., -1.], [-1., 1.]])
        cls.p = max_affine_partition(cls.data, cls.maf)

    def test_npoints(self):
        self.assertEqual(self.p.npoints, 6)

    def test_ncells(self):
        self.assertEqual(self.p.ncells, 3)

    def test_consistency(self):
        self.p.assert_consistency()

    def test_cells(self):
        self.assertEqual(len(self.p.cells), 3)
        np.testing.assert_array_equal(self.p.cells[0], np.array([0]))
        np.testing.assert_array_equal(self.p.cells[1], np.array([3, 4]))
        np.testing.assert_array_equal(self.p.cells[2], np.array([1, 2, 5]))

    def test_cell_sizes(self):
        self.assertEqual(self.p.cell_sizes(), (1, 2, 3))

    def test_cell_indices(self):
        np.testing.assert_array_equal(self.p.cell_indices(), np.array([0, 2, 2, 1, 1, 2]))

    def test_equality_same(self):
        self.assertTrue(self.p == max_affine_partition(self.data, self.maf))

    def test_equality_different(self):
        self.assertFalse(self.p == singleton_partition(self.data.shape[0]))

    def test_equality_reordered_cells(self):
        self.assertTrue(self.p == Partition(npoints=6, ncells=3,
                                            cells=((3, 4), (1, 2, 5), (0,))))


class TestSingletonPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = singleton_partition(5)

    def test_npoints(self):
        self.assertEqual(self.p.npoints, 5)

    def test_ncells(self):
        self.assertEqual(self.p.ncells, 5)

    def test_cells(self):
        self.assertEqual(self.p.cells, ([0], [1], [2], [3], [4]))

    def test_consistency(self):
        self.p.assert_consistency()

    def test_cell_sizes(self):
        self.assertEqual(self.p.cell_sizes(), (1, 1, 1, 1, 1))

    def test_cell_indices(self):
        np.testing.assert_array_equal(self.p.cell_indices(), np.array([0, 1, 2, 3, 4]))

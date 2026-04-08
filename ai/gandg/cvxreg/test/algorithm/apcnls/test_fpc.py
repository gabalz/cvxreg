import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.common.partition import cell_radii
from ai.gandg.cvxreg.algorithm.apcnls.fpc import (
    farthest_point_clustering, adaptive_farthest_point_clustering,
)


class TestFarthestPointClustering(unittest.TestCase):
    def test_small_example(self):
        X = np.array(
            [[1., 1.], [-1., 1.], [0., 1.],
             [-1.5, 0.5], [0.5, -1.], [1., 1.5],
             [-1., -1.], [-2., -1], [0., 1.]],
        )
        p = farthest_point_clustering(ncells=3, data=X)
        self.assertEqual(p.npoints, 9)
        self.assertEqual(p.ncells, 3)
        self.assertEqual(p.cells, ([0, 1, 2, 3, 5, 8], [6, 7], [4]))

    def test_clustered_data(self):
        set_random_seed(19)
        C = np.random.randn(5, 3)
        expected_C = np.array([
            [-1.03525403,  0.41239449,  0.70404776],
            [-0.12290513,  0.12498242, -1.43440465],
            [-0.05608202,  0.20520597, -1.44680874],
            [ 0.49377073,  0.22724934, -0.97081668],
            [-0.17539455, -1.07307641, -0.72220697],
        ])
        np.testing.assert_array_almost_equal(C, expected_C)
        X = C[np.random.randint(C.shape[0], size=(100,)), :]
        X += np.random.randn(*X.shape) * 0.1
        p = farthest_point_clustering(ncells=5, data=X)
        self.assertEqual(p.npoints, 100)
        self.assertEqual(p.ncells, 5)
        self.assertEqual([len(c) for c in p.cells], [31, 15, 22, 22, 10])
        expected_means = np.array([
            [-0.07134101,  0.16107547, -1.42839121],
            [-1.04114572,  0.41896789,  0.70097091],
            [-0.17329119, -1.03270566, -0.71894531],
            [ 0.47347632,  0.22806483, -0.96611127],
            [-0.0059361 ,  0.15846984, -1.59665994],
        ])
        np.testing.assert_array_almost_equal(
            np.vstack([np.mean(X[c, :], axis=0).T for c in p.cells]),
            expected_means)


class TestAdaptiveFarthestPointClustering(unittest.TestCase):
    def test_zero_data(self):
        X = np.zeros((7, 2))
        partition = adaptive_farthest_point_clustering(data=X)
        self.assertEqual(partition.ncells, 1)
        np.testing.assert_array_equal(partition.cells[0], [0, 1, 2, 3, 4, 5, 6])

    def test_randn_200_2(self):
        set_random_seed(19)
        X = np.random.randn(200, 2)
        partition = adaptive_farthest_point_clustering(data=X)
        self.assertEqual(partition.npoints, 200)
        self.assertEqual(partition.ncells, 15)
        self.assertEqual([len(c) for c in partition.cells],
                         [63, 1, 5, 3, 1, 5, 4, 3, 13, 17, 10, 14, 16, 23, 22])
        self.assertEqual([int(np.mean(c)) for c in partition.cells],
                         [107, 117, 110, 166, 71, 110, 137, 81, 107, 111, 86, 67, 83, 100, 81])
        self.assertAlmostEqual(
            np.round(max(cell_radii(X, partition)), decimals=4), 0.8883)

    def test_clustered_data_q2(self):
        set_random_seed(19)
        # consume the same RNG state as the doctest (200x2 block above runs first)
        np.random.randn(200, 2)
        C = np.random.randn(5, 3)
        expected_C = np.array([
            [-0.57681049, -1.0982188 , -1.77021275],
            [-0.29580036,  0.55481809, -1.40079008],
            [-0.47390752, -0.02820206, -0.79310281],
            [-0.5338925 ,  0.07452468,  0.08311392],
            [-0.84831213, -1.02133226,  0.38650518],
        ])
        np.testing.assert_array_almost_equal(C, expected_C)
        X = C[np.random.randint(C.shape[0], size=(100,)), :]
        X += np.random.randn(*X.shape) * 0.1
        partition = adaptive_farthest_point_clustering(data=X, q=2)
        self.assertEqual(partition.npoints, 100)
        self.assertEqual(partition.ncells, 5)
        self.assertEqual([len(c) for c in partition.cells], [15, 17, 22, 21, 25])
        expected_means = np.array([
            [-0.49366699, -0.04686255, -0.77307882],
            [-0.61269755, -1.07736845, -1.77314976],
            [-0.86476045, -0.98983781,  0.40252153],
            [-0.28370596,  0.52195337, -1.41375348],
            [-0.54848627,  0.05573469,  0.06377456],
        ])
        np.testing.assert_array_almost_equal(
            np.vstack([np.mean(X[c, :], axis=0).T for c in partition.cells]),
            expected_means)

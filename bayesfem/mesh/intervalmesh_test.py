import unittest
import numpy as np
import tensorflow as tf
from bayesfem.mesh import IntervalMesh

class TestIntervalMesh(unittest.TestCase):
    def test_interval_mesh_nodes(self):
        """
        Test the mesh gives the nodes we expect
        """
        points = np.array([0, 1, 2, 3, 4], dtype=np.float32)[:, np.newaxis]
        mesh = IntervalMesh(points)

        tf_nodes = tf.constant(points, dtype=tf.float32)

        assert all(tf.math.equal(tf_nodes, mesh.nodes))

    def test_interval_mesh_interpolation(self):
        """
        Test the interpolation operator works properly
        """
        # construct the initial mesh
        nodes = np.linspace(0., 1., 5)[:, np.newaxis]
        mesh = IntervalMesh(nodes)

        # construct the mesh using scipy linear interpolator
        from scipy.interpolate import interp1d
        u = tf.math.sin(tf.squeeze(mesh.nodes))

        index_points = np.random.uniform(size=15)[:, None]

        O = mesh.linear_interpolation_operator(index_points)
        O = tf.linalg.LinearOperatorFullMatrix(O)
        u_ = interp1d(tf.squeeze(mesh.nodes).numpy(), u.numpy())

        assert np.allclose(u_(index_points[:, 0]), O.matvec(u))


if __name__ == '__main__':
    unittest.main()

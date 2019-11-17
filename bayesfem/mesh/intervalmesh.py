
import numpy as np
import tensorflow.compat.v2 as tf
from bayesfem.mesh import Mesh

__all__ = ['IntervalMesh', ]

class IntervalMesh(Mesh):
    def __init__(self, nodes):
        nodes = tf.sort(nodes, axis=-2)
        super(IntervalMesh, self).__init__(nodes, dtype=nodes.dtype)

        elements = np.column_stack(
            [np.arange(0, self.n_nodes-1),
             np.arange(1, self.n_nodes)])

        self._boundary_node_indices = [0, self.n_nodes-1]
        self._interior_node_indices = np.arange(1, self.n_nodes-1)

        self._elements = elements
        self._element_type = 'Interval'

        self._element_volumes = tf.squeeze(self.nodes[1:] - self.nodes[:-1])

    def linear_interpolation_operator(self, index_points):
        """ Returns the linear operator that carries out interpolation of the solution at node points.

        Parameters
        ----------
        index_points : array, shape=[..., nobs, 1]
            points at which we want to interpolate

        Returns
        -------
        O : batched array, shape = [..., nobs, mesh.npoints]
            linear operator such that O @ u interpolates u[mesh.points] onto
            index_points


        .. note::

            The matrix of the operator returned will have 2 entries per row, and so
            will be sparse when `mesh.npoints >> index_points.shape[-1]`.

        """
        # we need to find which element each of index points is in
        # we exploit the fact that the interval mesh is order and then
        # broadcast >
        is_greater = tf.squeeze(
            index_points[:, tf.newaxis, :] > self.nodes[tf.newaxis, ...])

        # find the max along each row, corresponding to an index point, to find
        # the last interval [ta[i], tb[i]] for which index_point[m] > ta[i]
        index_point_elements = tf.argmax(
            tf.cast(is_greater, tf.int32) * tf.range(0, self.n_nodes, 1)[tf.newaxis, :],
            axis=-1)

        ta = tf.gather(self.nodes, index_point_elements)
        tb = tf.gather(self.nodes, index_point_elements + 1)

        index_point_element_volumes = tf.gather(
            self.element_volumes, index_point_elements)[:, tf.newaxis]

        # now we have found which element the index point is in we
        # evaluate the local basis functions on these elements
        phi1 = (tb - index_points) / index_point_element_volumes
        phi2 = (index_points - ta) / index_point_element_volumes

        # indices to update Oa
        #[*zip(np.arange(index_points.shape[-2]), index_point_elements)]
        Oa_indices = tf.concat([np.arange(index_points.shape[-2])[:, tf.newaxis],
                                index_point_elements[:, tf.newaxis]], axis=-1)
        Oa = tf.scatter_nd(Oa_indices,
                           tf.squeeze(phi1),
                           shape=[index_points.shape[-2], self.n_nodes])

        # incremenet the index_point_elements by one, taking advantage of the sorting
        Ob_indices = tf.concat([np.arange(index_points.shape[-2])[:, tf.newaxis],
                                (index_point_elements + 1)[:, tf.newaxis]],
                               axis=-1)
        Ob = tf.scatter_nd(Ob_indices,
                           tf.squeeze(phi2),
                           shape=[index_points.shape[-2], self.n_nodes])

        return Oa + Ob

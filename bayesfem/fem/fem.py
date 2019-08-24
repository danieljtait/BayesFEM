import tensorflow as tf
import numpy as np

__all__ = ['BaseFEM', ]


class BaseFEM:
    """ Base FEM object. """
    def __init__(self, mesh, name=None):
        """ Instatiates a BaseFEM object. """
        self._mesh = mesh
        self._dtype = mesh.points.dtype
        self._name = name
        self._batch_shape = tf.constant([], dtype=tf.int32)

    @property
    def name(self):
        """ Name of FEM problem. """
        return self._name

    @property
    def mesh(self):
        """ Underlying mesh on which the FEM is being solved. """
        return self._mesh

    @property
    def dtype(self):
        """ Common data dtype for the tensors comprising the solver. """
        return self._dtype

    @property
    def batch_shape(self):
        return self._batch_shape

    def _apply_dirchlet_bound_conditions(self, A, b):
        """

        Parameters
        ----------

        A : Tensor
            stiffness matrix

        b : Tensor
            load vector

        Returns
        -------

        A : Tensor

        b : Tensor
        """
        with tf.name_scope("ApplyBoundaryConditions") as scope:

            def boundary_correct_stiffness_matrix(A):
                """ applies the boundary correction to a single stiffness matrix. """
                boundary_inds = [[i, j] for i in self.mesh.boundary_nodes
                                 for j in range(self.mesh.npoints)]
                updates = tf.zeros(len(boundary_inds), dtype=self.dtype)

                A = tf.tensor_scatter_nd_update(A, boundary_inds, updates)

                # repeat for symmetry
                boundary_inds = [[j, i] for j in range(self.mesh.npoints)
                                 for i in self.mesh.boundary_nodes]
                updates = tf.zeros(len(boundary_inds), dtype=self.dtype)

                A = tf.tensor_scatter_nd_update(A, boundary_inds, updates)

                boundary_inds = [[i, i] for i in self.mesh.boundary_nodes]

                updates = tf.ones(len(boundary_inds), dtype=self.dtype)

                A = tf.tensor_scatter_nd_update(A, boundary_inds, updates)

                return A

            batch_shape = A.shape[:-2]
            if len(batch_shape) == 0:
                A = boundary_correct_stiffness_matrix(A)

            else:
                A = tf.map_fn(boundary_correct_stiffness_matrix, A)

            b = tf.tensor_scatter_nd_update(
                b,
                [[i] for i in self.mesh.boundary_nodes],
                tf.zeros(len(self.mesh.boundary_nodes),
                         dtype=self.dtype))

            return A, b

    def _get_quadrature_nodes(self):
        if self.mesh.element_type == 'Interval':
            # quadrature degree = 1
            quad_degree = 1

            if quad_degree == 1:
                nodes = self.mesh.points[:-1] + .5*np.diff(self.mesh.points)

                return nodes

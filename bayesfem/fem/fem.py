import tensorflow as tf

__all__ = ['BaseFEM', ]


class BaseFEM:
    """ Base FEM object. """
    def __init__(self, mesh,):
        """ Instatiates a BaseFEM object. """
        self._mesh = mesh
        self._dtype = mesh.points.dtype

    @property
    def mesh(self):
        """ Underlying mesh on which the FEM is being solved. """
        return self._mesh

    @property
    def dtype(self):
        """ Common data dtype for the tensors comprising the solver. """
        return self._dtype

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

            b = tf.tensor_scatter_nd_update(
                b,
                [[i] for i in self.mesh.boundary_nodes],
                tf.zeros(len(self.mesh.boundary_nodes),
                         dtype=self.dtype))

            return A, b

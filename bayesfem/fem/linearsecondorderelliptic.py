import numpy as np
import tensorflow as tf
from .fem import BaseFEM


class LinearSecondOrderElliptic(BaseFEM):
    """ Linear second order elliptic PDE solver using FEM """
    def __init__(self,
                 coeff,
                 source,
                 mesh,
                 name='LinearSecondOrderElliptic'):
        """ Instantiates a linear second order elliptic PDE solver. """
        super(LinearSecondOrderElliptic, self).__init__(mesh)

        with tf.name_scope('Init') as scope:
            self._coeff = coeff
            self._source = source

    @property
    def coeff(self):
        return self._coeff

    @property
    def source(self):
        """ Source function. """
        return self._source

    def assemble(self, coeff=None):
        """ Assembles the stiffness and load matrix. """
        with tf.name_scope('{}FEMAssemble'.format(self.name)):

            batch_shape = self.batch_shape if coeff is None else coeff.shape[:-1]
            stiffness_matrix_shape = tf.concat((batch_shape,
                                                [self.mesh.npoints, self.mesh.npoints]), axis=0)

            A = tf.zeros(stiffness_matrix_shape,
                         name='stiffness_matrix',
                         dtype=self.dtype)
            b = tf.zeros((self.mesh.npoints,),
                         name='load_vector',
                         dtype=self.dtype)

            if coeff is None:
                coeff = self.coeff

            local_stiffness = self._local_stiffness_matrix_calculator(coeff)
            local_load = local_load_calculator(self)

            # work out the shape for adding local
            # stiffness matrices into possibly batched global matrices
            local_stiffness_flat_shape = tf.concat(
                (batch_shape, [tf.reduce_prod(local_stiffness.shape[-2:])]),
                axis=0)

            for k, row in enumerate(self.mesh.elements):
                inds = [[i, j] for i in row for j in row]

                # inflate local values to their corresponding indices in the global stiffness matrix
                local_values = tf.reshape(local_stiffness[..., k, :, :], local_stiffness_flat_shape)

                if len(batch_shape) > 0:
                    inflated_local_values = tf.map_fn(
                        lambda x: tf.scatter_nd(inds, x, shape=[self.mesh.npoints, self.mesh.npoints]),
                        local_values
                    )
                else:
                    inflated_local_values = tf.scatter_nd(inds,
                                                          local_values,
                                                          shape=[self.mesh.npoints, self.mesh.npoints])


                # add the local values to global stiffnes matrix
                A += inflated_local_values

                local_load_values = tf.reshape(local_load[k], [-1])
                b = tf.tensor_scatter_nd_add(b,
                                             [[i] for i in row],
                                             local_load_values)
            return self._apply_dirchlet_bound_conditions(A, b)

    def _local_stiffness_matrix_calculator(self, coeff):
        return local_stiffness_matrix_calculator(self, coeff)


def local_stiffness_matrix_calculator(fem, coeff):
    if fem.mesh.element_type == 'Interval':
        """ One dimensional mesh with interval elements. 

        The local stiffness matrix is given by

            (ai / h) * | 1 -1|
                       |-1  1|
        """
        interval_lengths = np.diff(fem.mesh.points)

        E = np.array([[1., -1],
                      [-1., 1.]],
                     dtype=fem.dtype)

        if callable(coeff):
            a = coeff(.5 * (fem.mesh.points[:-1]
                            + fem.mesh.points[1:]))

        elif isinstance(coeff, (tf.Tensor, tf.Variable, np.ndarray)):
            a = coeff

        else:
            raise ValueError("`fem.coeff` must either be `callable` or "
                             "else an array of shape `[..., fem.mesh.get_quadrature_nodes().shape`")

        local_stiffness_matrix = (a / interval_lengths)[..., None, None] * E
        return local_stiffness_matrix


def local_load_calculator(fem):
    if fem.mesh.element_type == 'Interval':
        """ One dimensional mesh with interval elements.

        The local load vector is given by 

            (1 / 2) | f(x_i-1) |
                    | f(x_i)   |
        """

        nodes = np.zeros(1, dtype=fem.dtype)
        weights = 2 * np.ones(1, dtype=fem.dtype)

        h = np.diff(fem.mesh.points)
        a, b = fem.mesh.points[fem.mesh.elements].T

        def gquad(f, a, b):
            fvals = f(.5 * (b - a) * nodes + .5 * (a + b))
            return .5 * (b - a) * weights * fvals

        def psi0_(x):
            return fem.source(x) * (b - x) / h

        def psi1_(x):
            return fem.source(x) * (x - a) / h

        local_load = np.column_stack(
            [gquad(psi0_, a, b),
             gquad(psi1_, a, b)])

        return local_load

import tensorflow as tf
import numpy as np

from .fem import BaseFEM


class Poisson(BaseFEM):

    def __init__(self, mesh, name='Poisson'):
        """ Instatiates a Poisson FEM solver"""
        super(Poisson, self).__init__(mesh)

    def assemble(self):
        """ Assembles the stiffness and load matrix. """
        pass


class LinearSecondOrderElliptic(BaseFEM):
    """ Linear second order elliptic PDE solver using FEM """

    def __init__(self, mesh, name='LinearSecondOrderElliptic'):
        """ Instantiates a linear second order elliptic PDE solver. """
        super(LinearSecondOrderElliptic, self).__init__(mesh)

    #@tf.function
    def assemble(self, a):
        """

        Parameters
        ----------
        a : Tensor
            element mid points values of the coefficients

        Returns
        -------

        A : stiffness matrix, Tensor

        b : load vector, Tensor

        ToDo: This is currently set up entirely for R2 -- make it dimension independent
        """

        with tf.name_scope('FEMAssemble') as scope:

            x, y = self.mesh.points[self.mesh.elements].T
            triareas = .5 * ((x[0] - x[2])*(y[1] - y[0])
                             - (x[0]-x[1])*(y[2]-y[0]))

            # computation of the hat gradients
            b = .5 * np.column_stack([y[1] - y[2],
                                      y[2] - y[0],
                                      y[0] - y[1]]) / triareas[:, None]
            c = .5 * np.column_stack([x[2] - x[1],
                                      x[0] - x[2],
                                      x[1] - x[0]]) / triareas[:, None]

            bouter = b[..., None] * b[:, None, :]
            couter = c[..., None] * c[:, None, :]

            local_stiffness = (bouter + couter) * triareas[..., None, None]

            A = tf.zeros((self.mesh.npoints, self.mesh.npoints),
                         name='stiffness_matrix',
                         dtype=self.dtype)
            b = tf.zeros((self.mesh.npoints, ),
                         name='load_vector',
                         dtype=self.dtype)

            for k, row in enumerate(self.mesh.elements):

                ###
                # Assembly of stiffness matrix
                # global inds for local element
                inds = [[i, j] for i in row for j in row]

                # local values of the stiffness matrix
                local_values = local_stiffness[k].ravel() * a[k]

                A = tf.tensor_scatter_nd_add(A,
                                             inds,
                                             local_values,
                                             name='scatter_nd_to_global')

                ###
                # Assembly of load vector
                local_load = triareas[k] * tf.ones((3, ), dtype=self.dtype) / 3.

                b = tf.tensor_scatter_nd_add(b,
                                             [[i] for i in row],
                                             local_load)
            return self._apply_dirchlet_bound_conditions(A, b)

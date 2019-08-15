import tensorflow as tf
from .fem import BaseFEM


class Poisson(BaseFEM):

    def __init__(self, mesh, name='Poisson'):
        """ Instatiates a Poisson FEM solver"""
        super(Poisson, self).__init__(mesh)


    def assemble(self):
        """ Assembles the stiffness and load matrix. """



class LinearSecondOrderElliptic(BaseFEM):
    """ Linear second order elliptic PDE solver using FEM """

    def __init__(self, mesh, dtype=tf.float32, name='LinearSecondOrderElliptic'):
        """ Instantiates a linear second order elliptic PDE solver. """
        super(LinearSecondOrderElliptic, self).__init__(mesh, dtype=dtype)

    def assemble(self):

        with tf.name_scope('Assembly') as scope:

            A = tf.zeros((self.mesh.npoints, self.mesh.npoints), dtype=self.dtype)
            b = tf.zeros((self.mesh.npoints, ), dtype=self.dtype)

            for k, row in enumerate(self.mesh.elements):

                ###
                # Assembly of stiffness matrix
                # global inds for local element
                inds = [[i, j] for i in row for j in row]

                # local values of the stiffness matrix
                values = Alocal[k].ravel() * a[k]

                A = tf.tensor_scatter_nd_add(A,
                                             inds,
                                             values,
                                             name='scatter_nd_to_global')

                ###
                # Assembly of load vector
                local_load = triareas[k] * tf.ones((3, ), dtype=self.dtype) / 3.

                b = tf.tensor_scatter_nd_add(b,
                                             [[i] for i in row],
                                             local_load)

        return A, b


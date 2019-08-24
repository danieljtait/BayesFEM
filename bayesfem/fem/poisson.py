import tensorflow as tf
import numpy as np

from .fem import BaseFEM
from .linearsecondorderelliptic import LinearSecondOrderElliptic


class Poisson(LinearSecondOrderElliptic):

    def __init__(self,
                 source,
                 mesh,
                 name='Poisson'):
        """ Instatiates a Poisson FEM solver"""

        def coeff(x):
            return tf.ones([1], self.dtype)

        super(Poisson, self).__init__(coeff, source, mesh, name=name)


class LinearSecondOrderElliptic_(BaseFEM):
    """ Linear second order elliptic PDE solver using FEM """

    def __init__(self, mesh, name='LinearSecondOrderElliptic'):
        """ Instantiates a linear second order elliptic PDE solver. """
        super(LinearSecondOrderElliptic, self).__init__(mesh)

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


def local_stiffness_matrix_calculator(fem):
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

        if callable(fem.coeff):
            a = fem.coeff(.5*(fem.mesh.points[:-1]
                              + fem.mesh.points[1:]))

        elif isinstance(fem.coeff, (tf.Tensor, tf.Variable, np.ndarray)):
            a = fem.coeff

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
        weights = 2*np.ones(1, dtype=fem.dtype)

        h = np.diff(fem.mesh.points)
        a, b = fem.mesh.points[fem.mesh.elements].T

        def gquad(f, a, b):
            fvals = f(.5*(b-a)*nodes + .5*(a+b))
            return .5*(b-a)*weights*fvals

        def psi0_(x):
            return fem.source(x)*(b-x)/h

        def psi1_(x):
            return fem.source(x)*(x-a)/h

        local_load = np.column_stack(
            [gquad(psi0_, a, b),
             gquad(psi1_, a, b)])

        #local_load = (.5 * np.column_stack([fem.source(fem.mesh.points[:-1]),
        #                                    fem.source(fem.mesh.points[1:])])
        #              * np.diff(fem.mesh.points)[..., None])

        return local_load


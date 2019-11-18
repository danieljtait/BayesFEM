"""
FEM solver for the class of linear second order elliptic PDEs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import itertools

__all__ = ['LinearSecondOrderElliptic', ]


def _local_stiffness_matrix_assembler(fem, coeff):
    """ Assembly of the local stiffness matrices for linear second order elliptic PDEs. """
    try:
        element_type = fem.mesh.element_type

        if element_type == 'Triangular':
            # Two dimensional mesh with triangular elements
            #
            # The local stiffnes matrix is given by

            # get the X and Y corrdinates of each element in the order respecting
            # self.elements

            X, Y = [tf.reshape(
                tf.gather(fem.mesh.nodes[:, i],
                          [*fem.mesh.elements.flat]),
                fem.mesh.elements.shape) for i in range(2)]

            # compute the gradients of the hat functions
            # b.shape = [3, b1,..., bB, mesh.n_elements]
            b = tf.concat([col[..., None] for col in [Y[..., 1] - Y[..., 2],
                                                      Y[..., 2] - Y[..., 0],
                                                      Y[..., 0] - Y[..., 1]]],
                          axis=-1) / (2 * fem.mesh.element_volumes[..., :, None])

            c = tf.concat([col[..., None] for col in [X[..., 2] - X[..., 1],
                                                      X[..., 0] - X[..., 2],
                                                      X[..., 1] - X[..., 0]]],
                          axis=-1) / (2 * fem.mesh.element_volumes[..., :, None])

            E = (b[..., None] * b[..., None, :]
                 + c[..., None] * c[..., None, :]) * fem.mesh.element_volumes[..., None, None]

            if callable(coeff):
                a = coeff(fem.mesh.barycenters)

            elif isinstance(coeff, (tf.Tensor, tf.Variable, np.ndarray)):
                a = coeff

            else:
                raise ValueError("`fem.coeff` must either be `callable` or "
                                 "else an array of shape `[..., fem.mesh.get_quadrature_nodes().shape`")

            local_stiffness_matrix = a[..., None, None] * E

        elif element_type == 'Interval':
            interval_lengths = fem.mesh.element_volumes

            E = tf.constant([[1., -1.],
                             [-1., 1.]], dtype=fem.dtype)

            if callable(fem.coeff):
                a = fem.coeff(.5 * (fem.mesh.nodes[:-1] + fem.mesh.nodes[1:]))

            elif isinstance(fem.coeff, (tf.Tensor, tf.Variable, np.ndarray)):
                a = fem.coeff

            else:
                raise ValueError("`fem.coeff` must either be `callable` or "
                                 "else an array of shape `[..., fem.mesh.get_quadrature_nodes().shape`")

            local_stiffness_matrix = (a / interval_lengths)[..., None, None] * E

        return local_stiffness_matrix

    except AttributeError:
        raise AttributeError("Unrecognised mesh.")


def _local_load_vector_assembler(fem, source):
    try:
        element_type = fem.mesh.element_type

        if element_type == 'Triangular':
            X, Y = [tf.reshape(
                tf.gather(fem.mesh.nodes[:, i],
                          [*fem.mesh.elements.flat]),
                fem.mesh.elements.shape) for i in range(2)]

            if callable(source):
                f = source([X, Y])
            else:
                f = source

            # allows the source to be a scalar
            f = tf.broadcast_to(f, X.shape)
            f = f * fem.mesh.element_volumes[:, tf.newaxis] / 3

        elif element_type == 'Interval':
            nodes = tf.zeros([1], dtype=fem.dtype)
            weights = 2 * tf.ones([1], dtype=fem.dtype)

            h = fem.mesh.element_volumes[..., tf.newaxis]
            a, b = fem.mesh.nodes[:-1], fem.mesh.nodes[1:]

            def gquad(f, a, b):
                fvals = f(.5 * (b - a) * nodes + .5 * (a + b))
                return .5 * (b - a) * fvals * weights

            def psi0_(x):
                return source(x) * (b - x) / h

            def psi1_(x):
                return source(x) * (x - a) / h

            f = tf.concat(
                [gquad(psi0_, a, b),
                 gquad(psi1_, a, b)],
                axis=-1)
            f = tf.squeeze(f)

        return f


    except AttributeError:
        raise AttributeError("Unrecognised mesh.")


class BasePDEModel:

    @property
    def batch_shape(self):
        """ The batch dimensions are indexes into indepedent,
        distinctly parameterised versions, of this PDE model.

        Returns
        -------
          batch_shape: `TensorShape`, possibly unknown
        """
        return tf.TensorShape(self._batch_shape)


class LinearSecondOrderElliptic(BasePDEModel):
    """
    FEM solver for problems of the form

    .. math::

        -\\nabla \\cdot (a(x) \\nabla u(x) ) = f(x)

    where :math:`a(x)` is the "coefficient" function and :math:`f(x)`
    is referred to as the "source" function.

    """
    def __init__(self,
                 domain,
                 coeff=None,
                 source=None,
                 name='LinearSecondOrderElliptic'):
        """
        Parameters
        ----------

        mesh : bayesfem.mesh.Mesh object
            Mesh for the FEM problem.

        coeff : callable or tensor, default None
            Either a callable, or a tensor of shape [..., mesh.get_quadrature_nodes().shape[-2]]

        source : callable or tensor, default None
            Either a callable source function defined over the domain, or else a Tensor of
            shape `mesh.n_nodes`. Default `lambda x: tf.ones([1])`

        """
        self._domain = domain

        # default: constant unit coefficient function
        if coeff is None:
            coeff = lambda x: tf.ones([1], self.dtype)
        self._coeff = coeff

        # default: constant unit source function
        if source is None:
            source = lambda x: tf.ones([1], self.dtype)
        self._source = source

        # discover the batch shape
        if callable(self.coeff) and callable(self.source):
            batch_shape = []
        elif callable(self.coeff):
            # get the batch dimension from source
            batch_shape = self.source.shape[:-1]
        elif callable(self.source):
            batch_shape = self.coeff.shape[:-1]
        else:
            # both supplied as tensors
            # ToDo: Check broadcasting compatability
            batch_shape = self.coeff.shape[:-1]
        self._batch_shape = batch_shape

    def _local_stiffness_matrix_assembler(self, coeff=None):
        """ Assembles the local stiffness matrices.

        Parameters
        ----------

        coeff : Tensor or callable, default None
            Either a callable, or a tensor of shape [..., mesh.get_quadrature_nodes().shape[-2]]

        Returns
        -------

        local_stiffness_matrices : Tensor, shape [self.mesh.n_elements, elem_dim, elem_dim]
            Tensor of Local stifness matrices for each element, dimension depends on
            number of local basis elements.
        """
        if coeff is None:
            coeff = self.coeff
        return _local_stiffness_matrix_assembler(self, coeff)

    def _local_load_vector_assembler(self, source=None):
        if source is None:
            source = self.source
        return _local_load_vector_assembler(self, source)

    def _assemble(self, global_scatter=True, coeff=None):
        a_local_values = self._local_stiffness_matrix_assembler(coeff=coeff)
        b_local_values = self._local_load_vector_assembler()
        # Don't scatter into global matrices
        if not global_scatter:
            return a_local_values, b_local_values

        global_stiffness_indices = itertools.chain.from_iterable(
            [itertools.product(row, row) for row in self.mesh.elements])

        if len(self.batch_shape) == 0:
            # add an extra leading dimension onto a_local
            a_local_values = a_local_values[tf.newaxis, ...]
            batch_shape = [1]
        else:
            batch_shape = self.batch_shape

        # flatten a_local_values
        flat_shape = tf.concat(
            (batch_shape,
            [a_local_values.shape[-3]*a_local_values.shape[-1]**2]),
            axis=0)

        a_local_values = tf.reshape(a_local_values, flat_shape,
                                    name='flatten_local_stiffness_matrix')

        # unpack global_stiffness_indices
        global_stiffness_indices = [*global_stiffness_indices]
        A = tf.map_fn(
            lambda x: tf.scatter_nd(
                global_stiffness_indices, x,
                shape=[self.mesh.n_nodes, self.mesh.n_nodes]),
            a_local_values)
        A = tf.squeeze(A)

        global_load_indices = itertools.chain.from_iterable(
            [zip(row, np.zeros(3, dtype=np.intp)) for row in self.mesh.elements])

        b = tf.scatter_nd([*global_load_indices],
                          tf.reshape(b_local_values, [-1],
                                     name='flatten_local_load'),
                          shape=[self.mesh.n_nodes, 1],
                          name='scatter_nd_to_global_load')

        return A, b

    def _solve_dirichlet(self):
        # check the boundary conditions are correct
        if self.domain.boundary.boundary_condition_type != 'Dirichlet':
            raise ValueError("_solve_dirichlet must be used with Dirichlet boundary conditions.")

        A, b = self._assemble()

        # work out if any batching needs to be done
        if len(self.batch_shape) == 0:
            # add an extra leading dimension onto A
            A = A[tf.newaxis, ...]
            batch_shape = [1] # false batch shape, squeezed out by end
        else:
            batch_shape = self.batch_shape

        # get the interior of A
        global_stiffness_interior_indices = [*itertools.product(
            self.mesh.interior_node_indices, repeat=2)]

        Ainterior = tf.map_fn(
            lambda x: tf.reshape(
                tf.gather_nd(x, global_stiffness_interior_indices),
                [len(self.mesh.interior_node_indices),
                 len(self.mesh.interior_node_indices)]), A)

        b_interior = tf.gather_nd(
            b, [*zip(self.mesh.interior_node_indices,
                     [0]*len(self.mesh.interior_node_indices))])

        interior_bound_indices = [*itertools.product(
            self.mesh.interior_node_indices,
            self.mesh.boundary_node_indices)]

        Aint_bnd = tf.map_fn(
            lambda x: tf.reshape(
                tf.gather_nd(x, interior_bound_indices),
                [len(self.mesh.interior_node_indices),
                 len(self.mesh.boundary_node_indices)]), A)


        bnd_node_indices = np.array(self.mesh.boundary_node_indices, dtype=np.intp)
        int_node_indices = np.array(self.mesh.interior_node_indices, dtype=np.intp)

        # get the value on the boundary
        g = self.domain.boundary.g

        # convert the stiffness matrices to operators for batch matmul
        Ainterior_op = tf.linalg.LinearOperatorFullMatrix(Ainterior)
        Aint_bnd_op = tf.linalg.LinearOperatorFullMatrix(Aint_bnd)

        # add the fixed dirichlet conditions to sol
        # ToDo: Batch boundary values
        sol = tf.scatter_nd(bnd_node_indices[:, None],
                            g,
                            shape=[self.mesh.n_nodes, 1])

        b_ = b_interior[..., tf.newaxis] - Aint_bnd_op.matmul(g)
        sol_interior = Ainterior_op.solve(b_)

        # sol_interior has a batched shape [b, n_interior_nodes, 1]
        return tf.squeeze(tf.map_fn(
            lambda x: tf.tensor_scatter_nd_add(
                sol, int_node_indices[:, None], x),
            sol_interior))[..., tf.newaxis]  # kills pesduo-batch dimensions, but keeps output a vector

    @property
    def dtype(self):
        """ Common data type of the mesh nodes, coefficient and source function values. """
        return self._dtype

    @property
    def coeff(self):
        """ The coefficient function of the Elliptic PDE. """
        return self._coeff

    @property
    def source(self):
        """ The source function of the Elliptic PDE. """
        return self._source

    @property
    def domain(self):
        return self._domain

    @property
    def mesh(self):
        """ Mesh for the FEM solver.

        Returns
        -------

        mesh : bayesfem.fem.Mesh object
            Mesh for the FEM solver.
        """
        return self.domain.mesh

    @property
    def dtype(self):
        """ Data type of the mesh

        .. todo::
            It should in principle be possible to have different data types
            for the mesh nodes -- i.e. the inputs to the source, coeff function
            -- and the the outputs of these functions, which determine the
            dtype of the FEM solution.
        """
        return self.mesh.dtype

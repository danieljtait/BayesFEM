import tensorflow as tf
import numpy as np
import itertools


def boundary_zero(A, mesh):
    """ Zeros all rows and columns of A with a node on the boundary """

    dtype = A.dtype

    boundary_inds = [[i, j] for i in mesh.boundary_nodes
                     for j in range(mesh.npoints)]
    updates = tf.zeros(len(boundary_inds), dtype=dtype)

    A = tf.tensor_scatter_nd_update(A, boundary_inds, updates)

    # repeat for symmetry
    boundary_inds = [[j, i] for j in range(mesh.npoints)
                     for i in mesh.boundary_nodes]
    updates = tf.zeros(len(boundary_inds), dtype=dtype)

    A = tf.tensor_scatter_nd_update(A, boundary_inds, updates)

    return A


def boundary_ones_add_op(mesh):
    """ Creates a matrix with `M[i, i] = 1.0` if i is a boundary node

    ToDo: Consider changing return type to `tf.linalg.LinearOperator`.

    Parameters
    ----------

    mesh : vbfem.mesh.Mesh object
        Mesh of the FEM problem

    Returns
    -------

    boundary_ones : Tensor
        Tensor with [..., i, i] = 1.0 if i in `mesh.boundary_nodes`

    """
    boundary_indices = [[i, i] for i in mesh.boundary_nodes]
    updates = tf.ones(len(boundary_indices), dtype=mesh.points.dtype)
    return tf.scatter_nd(boundary_indices,
                         updates,
                         shape=[mesh.npoints, mesh.npoints])


def scatter_matrixprod(indices, updates, shape, B):
    n = B.shape[-2]
    range_n = range(n)

    inds_ = np.concatenate([[*zip([i]*n, range_n)]
                            for i, _ in indices])
    updates_ = tf.concat([updates[k] * B[j, :]
                          for k, (_, j) in enumerate(indices)], axis=0)
    return tf.scatter_nd(inds_, updates_, shape)


def scatter_matrixquad_old_nslow(indices1, updates1,
                       indices2, updates2,
                       Q, shape):
    """ Efficient evaluation of A @ Q @ B when A and B are formed
    by tf_scatter

    :param indices1:
    :param updates1:
    :param indices2:
    :param updates2:
    :param Q:
    :param shape:
    :return:
    """
    indices = []
    updates = []
    for n, (i, j) in enumerate(indices1):
        for m, (i_, j_) in enumerate(indices2):
            indices.append([i, j_])
            updates.append(Q[j, i_] * updates1[n] * updates2[m])
    return tf.scatter_nd(indices, updates, shape)


def scatter_matrixquad(indices1, updates1,
                       indices2, updates2,
                       Q, shape):
    # inds we want from Q are Q[CartesianProduct(indices1[:, 1], indices2[:, 0] )]
    q = tf.gather_nd(Q, [*itertools.product(indices1[:, 1],
                                            indices2[:, 0])])

    updates = (tf.reshape(updates1[:, None] * updates2[None, :], [-1])
               * q)
    indices = [*itertools.product(indices1[:, 0], indices2[:, 1])]

    return tf.scatter_nd(indices,
                         updates,
                         shape)


def get_E_AQA(mean, cov, vbfem, Q):

    var = tf.linalg.diag_part(cov)

    expec_local_stiffness = vbfem._local_stiffness_matrix_calculator(
        tf.exp(mean + .5*var))
    # flatten the local stiffness matrix
    expec_local_stiffness = tf.map_fn(lambda x: tf.reshape(x, [-1]),
                                      expec_local_stiffness)

    result = tf.zeros([1], dtype=vbfem.dtype)

    # also efficient calculate this expectation
    expec_AQ =tf.zeros([1], dtype=vbfem.dtype)

    for k, row_k in enumerate(vbfem.mesh.elements):

        inds_k = np.array([[i, j] for i in row_k for j in row_k])

        #updates_k = tf.reshape(
        #    expec_local_stiffness[k], [-1])

        updates_k = expec_local_stiffness[k]

        expec_AQ += scatter_matrixprod(inds_k,
                                       updates_k,
                                       [vbfem.mesh.npoints,
                                        vbfem.mesh.npoints],
                                       Q)
        for l, row_l in enumerate(vbfem.mesh.elements):

            inds_l = np.array([[i, j] for i in row_l for j in row_l])

            #updates_l = tf.reshape(
            #    expec_local_stiffness[l], [-1])
            updates_l = expec_local_stiffness[l]

            cov_kl = tf.squeeze(tf.gather_nd(cov, [[k, l], ]))

            incr = scatter_matrixquad(
                inds_k,
                updates_k * tf.exp(.5*cov_kl),
                inds_l,
                updates_l * tf.exp(.5*cov_kl),
                Q,
                [vbfem.mesh.npoints, vbfem.mesh.npoints]
            )

            result += incr

    return expec_AQ, result


def get_EAQA(mean, cov, vbfem, Q):
    """ Calculates E[A @ Q] and E[A @ Q @ A] where A is the stiffness matrix
    corresponding to coefficient function `exp(z) z ~ N(mean, cov`.

    ToDo: Rewrite to handle batched mean and covariances

    Parameters
    ----------

    mean : Tensor
        Mean of log diffusion coefficient.

    cov : Tensor
        Covariance of log diffusion coefficient.

    vbfem : `VariationalBayesFiniteElementMethod` object
        Variational bayes FEM solver.

    Q : Tensor

    Returns
    -------

    EAQ : Tensor
        Calculated value of E[A @ Q]

    EAQA : Tensor
        Calculated value of E[A @ Q @ A]

    """

    indices = np.array([[*itertools.product(row, row)]
                         for row in vbfem.mesh.elements])

    # indices of the values to be collected from Q
    q_gather_inds = itertools.chain.from_iterable(
        [itertools.product(ind_k_1, ind_l_0)
         for ind_k_1 in indices[..., 1]
         for ind_l_0 in indices[..., 0]])

    # indices we will update to
    update_inds = itertools.chain.from_iterable(
        [itertools.product(ind_k_0, ind_l_1)
         for ind_k_0 in indices[..., 0]
         for ind_l_1 in indices[..., 1]])

    # elements of the local stiffness matrix evalauted with
    # coefficient `E[exp(z)] z ~ N(mean, cov)`
    a = vbfem._local_stiffness_matrix_calculator(
        tf.exp(mean + .5*tf.linalg.diag_part(cov)))

    # no. of nodes defining each element
    element_dim = vbfem.mesh.elements.shape[-1]
    a = tf.reshape(a, [vbfem.mesh.n_elements, element_dim**2])

    # batches [i, j, ...] = outer(a[i, :], a[j, :])
    a_prod = a[:, None, :, None] * a[None, :, None, :]
    a_prod *= tf.exp(cov)[..., None, None]
    a_prod = tf.reshape(a_prod, [-1])

    EAQA = tf.scatter_nd([*update_inds],
                         a_prod * tf.gather_nd(Q, [*q_gather_inds]),
                         [vbfem.mesh.npoints, vbfem.mesh.npoints])

    with tf.name_scope('EAQ') as scope:
        n = Q.shape[-1]
        EAQ_update_inds = itertools.chain.from_iterable(
            [itertools.product(ind_k_0, range(n))
             for ind_k_0 in indices[..., 0]])

        EAQ_q_gather_inds = itertools.chain.from_iterable(
            [itertools.product(ind_k_1, range(n))
             for ind_k_1 in indices[..., 1]])

        EAQ_updates = (tf.gather_nd(Q, [*EAQ_q_gather_inds])
                       * tf.reshape(a[..., None]
                                    * tf.ones(n, dtype=vbfem.dtype),
                                    [-1]))

        EAQ = tf.scatter_nd([*EAQ_update_inds],
                            EAQ_updates,
                            shape=[vbfem.mesh.npoints,
                                   vbfem.mesh.npoints])

    return EAQ, EAQA

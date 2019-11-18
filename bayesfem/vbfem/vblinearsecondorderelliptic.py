from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import bayesfem
import itertools

__all__ = ['VariationalBayesLinearSecondOrderElliptic', ]


class VariationalBayesLinearSecondOrderElliptic(
    bayesfem.fem.LinearSecondOrderElliptic):
    """
    Class description.
    """
    def __init__(self, *args, **kwargs):
        super(VariationalBayesLinearSecondOrderElliptic, self).__init__(
            *args, **kwargs)

    def mean_moment_constraint(selef, qu, qu_z, qz):
        """ Calculates E[A[z] @ u ] under the variational distribution.
        """
        pass

    def covar_moment_constraint(self, qu_z, qz):
        """ Calculates Cov(A[z]@ u) = E[A Cov(u | z) A.T]
        under the variational distribution.

        Parameters
        ----------
            qu_z : tfp.distributions.Distribution object
                Conditional distribution of the state given log-diffusion coeff.

            qz : tfp.distributions.Distribution object
                Conditional distribution of the log-diffusion coeff process.

        Returns
        -------
            cov_Az_u : Tensor, shape = [mesh.n_nodes, mesh.n_nodes]
                Covariance matrix of the vector A[z] @ u under the variational
                distribution.
        """
        # Calculated E[e^{zn + zm}]
        # - zn + zm ~ N(z_mean[n] + z_mean[m], )
        z_covar = qz._covariance()
        z_mean = qz.loc

        zn_var = tf.linalg.diag_part(z_covar)
        E_exp_zn_plus_zm = tf.exp(
            z_mean[..., tf.newaxis] + z_mean[..., tf.newaxis, :]
            + .5*(zn_var[..., tf.newaxis] + zn_var[..., tf.newaxis, :]
                  + 2*z_covar))

        # get the conditional covariance of u given z
        ucondz_covar = qu_z._covariance()

        # get the local values
        a_local, _ = self._assemble(global_scatter=False,
                                    coeff=tf.ones([1], self.dtype))
        # flatten a_local last two dimensions
        a_local_flat_shape = [self.mesh.n_elements, self.mesh.element_dim**2]
        a_local = tf.reshape(a_local,
                             a_local_flat_shape)

        # the assembly of the stiffness matrix loops like
        # for row_n in rows:
        #     for row_m in rows:
        #         for r, (i, j) in enumerate(product(row_n))
        #             for s, (k, l) in enumerate(product(row_m))
        # so we need to gather all (j, l) in ucondz_covar
        condz_covar_gather_inds = []
        update_indices = []
        for row_n in self.mesh.elements:
            for row_m in self.mesh.elements:
                for i, j in itertools.product(row_n, row_n):
                    for (k, l) in itertools.product(row_m, row_m):
                        condz_covar_gather_inds.append((j, l))
                        update_indices.append((i, k))

        # now get local updates E[e^{zn + zm}] * ar * as * Cuz[j, l]
        # - a_local.shape = [..., n_elements, elem_dim**2]
        E_exp_zn_plus_zm = E_exp_zn_plus_zm[..., tf.newaxis, tf.newaxis]
        updates = (a_local[..., tf.newaxis, :, tf.newaxis]
                   * a_local[..., tf.newaxis, :, tf.newaxis, :]) * E_exp_zn_plus_zm

        updates = (tf.reshape(updates, [-1])
                   * tf.gather_nd(ucondz_covar, condz_covar_gather_inds))

        return tf.scatter_nd(update_indices, updates,
                             shape=[self.mesh.n_nodes, self.mesh.n_nodes])

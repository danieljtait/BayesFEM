import tensorflow as tf
import tensorflow_probability as tfp
from bayesfem.fem import LinearSecondOrderElliptic
from .vbfem import VariationalBayesBaseFEM
from .momentutil import get_E_AQA
from bayesfem.fem.boundary_util import boundary_util
import bayesfem

tfd = tfp.distributions


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def _partition_covar(cov, n1, n2):
    c1, c2 = tf.split(cov, [n1, n2], axis=-1)
    c11, c21 = tf.split(c1, [n1, n2], axis=-2)
    c12, c22 = tf.split(c2, [n1, n2], axis=-2)
    return c11, c12, c21, c22


class FullVariationalBayesLinearSecondOrderElliptic(LinearSecondOrderElliptic, VariationalBayesBaseFEM):

    def __init__(self,
                 coeff,
                 source,
                 mesh,
                 index_points=None,
                 name='VariationalBayesLinearSecondOrderElliptic'):
        """ Instantiate a VariationalBayesLinearSecondOrderElliptic object. """
        LinearSecondOrderElliptic.__init__(self, coeff, source, mesh)

        self._index_points = index_points

    def fit(self, x,
            y,
            observation_noise_variance,
            log_diff_coeff_prior,
            covar_structure='LowRankPlusDiagonal'):

        #x, covar_shape, par_split = _prepare_args(x, self, covar_structure)

        # useful shapes and dimensions
        # - full size: N of node + N elements
        Nfull = self.mesh.npoints + self.mesh.n_elements
        covar_shape = [Nfull, Nfull]
        par_split = [Nfull, Nfull**2]

        @tf.function
        def objective_function(x):
            mean, chol = tf.split(x, par_split, axis=0)
            chol = tf.linalg.LinearOperatorLowerTriangular(
                tf.reshape(chol, covar_shape))

            cov = chol.matmul(chol, adjoint_arg=True).to_dense()

            # partition cov
            Cuu, Cuz, Czu, Czz = _partition_covar(cov,
                                                  self.mesh.npoints,
                                                  self.mesh.n_elements)

            Lzz = tf.linalg.cholesky(Czz)

            # conditional covariance of u given z
            Cu_z = Cuu - Cuz @ tf.linalg.cholesky_solve(Lzz, Czu)

            mu, mz = tf.split(mean, [self.mesh.npoints, self.mesh.n_elements])

            # component of the density that depends only on the marginal of u

            ######################################
            #                                    #
            # E[-log p(y | u)] - E[ H(q(u|z)) ]  #
            #                                    #
            ######################################
            obs_op = self.mesh.linear_interpolation_operator(self.index_points)
            obs_opt_obs_op = tf.matmul(obs_op, obs_op, adjoint_a=True)

            tr_oto_covu = tf.reduce_sum(obs_opt_obs_op * Cuu)
            z = y[..., None] - obs_op @ mu[:, None]
            expec_log_obs_pdf = -.5 * (tr_oto_covu
                                       + tf.reduce_sum(tf.matmul(z, z, adjoint_a=True))) / observation_noise_variance

            # this should have the conditioned mean -- but we only take the entropy
            # out of it so the mean doesn't matter
            qu_z = tfd.MultivariateNormalFullCovariance(
                loc=mu, covariance_matrix=Cu_z)

            ######################
            #                    #
            # Dkl(q(z) || p(z) ) #
            #                    #
            ######################
            qz = tfd.MultivariateNormalTriL(
                loc=mz, scale_tril=Lzz)

            dkl = qz.kl_divergence(log_diff_coeff_prior.get_marginal_distribution())

            return -expec_log_obs_pdf - qu_z.entropy() + dkl

        @tf.function
        def constraint(x):
            mean, chol = tf.split(x, par_split, axis=0)
            # lazily Monte-Carlo the constraint
            quz = tfd.MultivariateNormalTriL(
                loc=mean, scale_tril=tf.reshape(chol, covar_shape))
            samples = tf.squeeze(quz.sample(100, seed=1234))

            u, z = tf.split(samples,
                            [self.mesh.npoints, self.mesh.n_elements], axis=-1)

            A, f = self.assemble(tf.exp(z))
            A = tf.linalg.LinearOperatorFullMatrix(A)
            Au = tf.squeeze(A.matmul(u[..., None]))
            EAu = tf.reduce_mean(Au, axis=0)

            ####
            # E[A(z)u] = E[A(z) E[u | z]]
            ##
            c1 = tf.squeeze(EAu) - f

            ###
            # Tr(Cov(Au))
            #
            cov_Au = tfp.stats.covariance(Au)
            chol_cov_Au = tf.linalg.cholesky(cov_Au)

            c2 = tf.reshape(chol_cov_Au, [-1])

            return c1, c2

        f = objective_function(x)

        c1, c2 = constraint(x)

        penalty = 0.0001
        return f + .5 * (tf.reduce_sum(c1**2) + tf.reduce_sum(c2**2)) / penalty


def _prepare_args(x,
                  vbfem,
                  covar_structure):

    if covar_structure == 'LowRankPlusDiagonal':
        # parameters [mean_u,
        #             mean_logdiff_coeff,
        #             covar_diag,
        #             lowrank_factor]
        pass

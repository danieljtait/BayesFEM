import tensorflow as tf
import itertools
from bayesfem.fem import BaseFEM

__all__ = ['VariationalBayesBaseFEM',]


class VariationalBayesBaseFEM(BaseFEM):
    """ Base class for Variational Bayesian FEM solver. """
    def __init__(self,
                 mesh):
        """ Instantiates a VariationalBayesBaseFEM object. """
        super(VariationalBayesBaseFEM, self).__init__(mesh)

    @property
    def index_points(self):
        """ Index points of observations. """
        return self._index_points

    @property
    def observation_noise_variance(self):
        return self._observation_noise_variance

    def foo(self):
        """ Special method that VB FEM objects have. """
        print("I'm far superior to MCMC!")

    def handle_zero_dirichlet_boundary_conditions(self, mean_u, cov_u):
        """

        Parameters
        ----------
        mean_u : Tensor
            Mean of the variational distribution Q(u)

        cov_u : Tensor
            Covariance of the variational distribution Q(u)

        Returns
        -------
        mean_u : Tensor
            Mean of the variational distribution inflated by boundary values

        cov_u : Tensor
            Covariance of the variational distribution padded with zeros for boundary values
        """

        interior_inds = self.mesh.interior_node_indices()

        mean_u_complete = tf.scatter_nd(interior_inds[:, None],
                                        mean_u,
                                        shape=[self.mesh.npoints])

        cov_u_complete = tf.scatter_nd(
            list(itertools.product(interior_inds, interior_inds)),
            tf.reshape(cov_u, [-1]),
            shape=[self.mesh.npoints, self.mesh.npoints])

        return mean_u_complete, cov_u_complete


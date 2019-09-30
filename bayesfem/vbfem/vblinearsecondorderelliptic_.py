import tensorflow as tf
import tensorflow_probability as tfp
import itertools
from bayesfem.fem import LinearSecondOrderElliptic

from .vbfem import VariationalBayesBaseFEM
from bayesfem.vbfem import momentutil

tfd = tfp.distributions


def _add_diagonal_shift(matrix, shift):
    return tf.linalg.set_diag(
        matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def _partition_covar(cov, n1, n2):
    c1, c2 = tf.split(cov, [n1, n2], axis=-1)
    c11, c21 = tf.split(c1, [n1, n2], axis=-2)
    c12, c22 = tf.split(c2, [n1, n2], axis=-2)
    return c11, c12, c21, c22


class VariationalBayesLinearSecondOrderElliptic_(LinearSecondOrderElliptic, VariationalBayesBaseFEM):

    def __init__(self,
                 coeff,
                 source,
                 mesh,
                 index_points=None,
                 name='VariationalBayesLinearSecondOrderElliptic'):
        """ Instantiate a VariationalBayesLinearSecondOrderElliptic object. """
        LinearSecondOrderElliptic.__init__(self, coeff, source, mesh)

        self._index_points = index_points

    def analytic_moments(self, mean, cov):
        mu, mz = tf.split(
            mean, [self.mesh.npoints - self.mesh.nboundary_nodes, self.mesh.n_elements])

        # partition the complete covariance
        Cuu, Cuz, Czu, Czz = _partition_covar(cov,
                                              self.mesh.npoints - self.mesh.nboundary_nodes,
                                              self.mesh.n_elements)
        mu, Cuu = self.handle_zero_dirichlet_boundary_conditions(mu, Cuu)

        # compute ... with [i, k] = E[exp(z[i]) * z[k]]
        mgf_arg = (mz + 0.5*tf.linalg.diag_part(Czz))[..., None]
        E_mgf = tf.exp(mgf_arg)  # = E[exp(z)]
        E_expz_z = (mz[None, :] + Czz) * tf.exp(mgf_arg)

        # marginal variance of z
        zz_var = tf.linalg.diag_part(Czz)
        E_expz_u = (E_mgf * (mu[..., None, :]
                             - (Czu/zz_var[..., None])*mz[..., None])
                    + (Czu/zz_var[..., None])*tf.linalg.diag_part(E_expz_z)[..., None])

        ###
        # Finally put it together using the local stiffness matrix
        E_Az_u = tf.zeros([1], dtype=self.dtype)
        local_stiffness_matrix = self._local_stiffness_matrix_calculator(
            tf.ones([1], self.dtype))
        # flatten the last two axes of local_stiffness_matrix
        local_stiffness_matrix_flat_shape = tf.concat(
            [local_stiffness_matrix.shape[:-2],
             [tf.reduce_prod(local_stiffness_matrix.shape[-2:])]],
            axis=0)
        local_stiffness_matrix = tf.reshape(local_stiffness_matrix,
                                            local_stiffness_matrix_flat_shape)

        for n, row in enumerate(self.mesh.elements):
            inds = [(i, j) for i in row for j in row]

            updates = local_stiffness_matrix[..., n, :]
            An = tf.scatter_nd(inds,
                               updates,
                               shape=[self.mesh.npoints, self.mesh.npoints])
            E_Az_u += tf.matmul(An, E_expz_u[n, :][..., None], a_is_sparse=True)

        E_Az_u = tf.squeeze(E_Az_u)
        # Apply boundary conditions
        boundary_values = tf.zeros([len(self.mesh.boundary_nodes)],
                                  dtype=self.dtype)
        boundary_node_inds = self.mesh.boundary_nodes
        E_Az_u = tf.tensor_scatter_nd_update(
            E_Az_u,
            [[i] for i in boundary_node_inds],
            boundary_values)

        return E_Az_u

    def anal_moments(self, mean, cov, nsamples):
        E_AuuTA = _get_EAuuTA(mean, cov, self, nsamples)


        # also attempt in by MC
        q = tfd.MultivariateNormalFullCovariance(mean, cov)
        samples = tf.squeeze(q.sample(nsamples))
        u_samples, z_samples = tf.split(samples, [self.mesh.npoints, self.mesh.n_elements], axis=-1)
        A_mc, _ = self.assemble(tf.exp(z_samples))
        A_mc = tf.linalg.LinearOperatorFullMatrix(A_mc)
        Au_mc = A_mc.matvec(u_samples)
        AuuA_mc = Au_mc[..., None] * Au_mc[..., None, :]

        return E_AuuTA, tf.reduce_mean(AuuA_mc, axis=0)

    def constraint(self, mean, cov, f):
        with tf.name_scope("Constraints") as scope:
            EAu = self.analytic_moments(mean, cov)
            c1 = EAu - f
            c2 = tf.zeros([1], self.dtype)

            # CovAzu = EAzuuTA - (EAzu)(EAzu).T
            # ... but we can apply constraint 1. to give
            # CovAzu = EAzuuTA - ff.T
            #EAuuTA = self.anal_moments(mean, cov, 1)
            #ffT = tf.matmul(f[:, None], f[None, :])
            #CovAzu = EAuuTA - ffT
            #c2 = tf.linalg.cholesky(_add_diagonal_shift(CovAzu, 10.))
            #c2 = tf.reshape(c2, [-1])

            return c1, c2

    def _calculate_constraints(self, mean, covariance_matrix):
        """

        Parameters
        ----------

        mean : Tensor
            Full mean for the var. Bayes. approximation

        covariance_matrix : Tensor
            Full covariance matrix for the var. Bayes. approximation

        Returns
        -------

        CovAu : Tensor
            Covariance matrix of `A[z] @ u` for the latent state `z` and output variable `u`
            under the variational approximation.

        """

        Cuu, Cuz, Czu, Czz = _partition_covar(covariance_matrix,
                                              self.mesh.npoints - self.mesh.nboundary_nodes,
                                              self.mesh.n_elements)

        mu, mz = tf.split(mean,
                          [self.mesh.npoints - self.mesh.nboundary_nodes,
                           self.mesh.n_elements],
                          axis=-1)

        # Handle the boundary condition by inflating the
        # mean and covariance matrices
        mu, Cuu = self.handle_zero_dirichlet_boundary_conditions(mu, Cuu)
        # ... also need to inflate the cross covariance, Cuz
        # ToDo: Move the cross-cov method to the general handle boundary conditions method

        interior_inds = self.mesh.interior_node_indices()
        Cuz = tf.scatter_nd(list(itertools.product(interior_inds,
                                                   list(range(self.mesh.n_elements)))),
                            tf.reshape(Cuz, [-1]),
                            [self.mesh.npoints, self.mesh.n_elements])

        # pre-calculate what we can before looping starts

        # i.) Moments
        # extract the marginal variance vector
        var_zn = tf.linalg.diag_part(Czz)

        # Calculate E[exp( z[n] + z[m])]
        exp_zn_plus_zm = tf.exp(mz[..., None] + mz[..., None, :]
                                + .5*(var_zn[..., None] + var_zn[..., :, None] + 2*Czz))

        # ... and E[z[n] * exp(z[n] + z[m])]
        zn_exp_zn_plus_zm = mz[..., None] + Czz + var_zn[..., None]
        zn_exp_zn_plus_zm *= tf.exp(mz[..., None] + mz[..., None, :]
                                    + .5 * (1 + Czz / var_zn[..., None]) ** 2 * var_zn[..., None]
                                    + .5 * (var_zn[..., None, :] - Czz ** 2 / var_zn[..., None]))

        # ... and also E[z[n]*z[m] * exp(z[n] + z[m])]
        znzm_exp_zn_plus_zm = (Czz +
                               (mz[..., None] + var_zn[..., None] + Czz)*(mz[..., None, :] + var_zn[..., None, :] + Czz))
        znzm_exp_zn_plus_zm *= exp_zn_plus_zm

        # ... and finally E[z[n]**2 * exp(z[n] + z[m])]
        _expr1 = var_zn[..., None] + (mz[..., None] + var_zn[..., None] + Czz) ** 2
        _expr2 = tf.exp(
            mz[..., None] + mz[..., None, :]
            + .5 * (var_zn[..., None, :] - Czz ** 2 / var_zn[..., None])
            + .5 * (1 + Czz / var_zn[..., None]) ** 2 * var_zn[..., None])

        znsq_exp_zn_plus_zm = _expr1 * _expr2

        # ii.) local values of the stiffness matrix
        local_stiffness = self._local_stiffness_matrix_calculator(
            tf.ones([1], self.dtype))
        # flatten the local stiffness matrix
        local_stiffness = tf.map_fn(lambda x: tf.reshape(x, [-1]), local_stiffness)

        # useful dimensions
        element_dim = self.mesh.elements.shape[-1]

        # initialise the result
        EAuuTA = tf.zeros([1], self.dtype)

        # Now loop!
        for n, row_n in enumerate(self.mesh.elements):

            # local indices
            In = [*itertools.product(row_n, repeat=2)]

            # gather mean of u_j for (, j) in In
            mean_uj = tf.gather_nd(mu, [[j, ] for (_, j) in In])

            for m, row_m in enumerate(self.mesh.elements):
                Im = [*itertools.product(row_m, repeat=2)]

                # Gather the mean of uk for (k, _) in Im
                mean_uk = tf.gather_nd(mu, [[k, ] for (k, _) in Im])

                # Calculate the covariance of u[j] and u[k] given z[n] and z[m]
                Cuj_znzm = tf.reshape(tf.gather_nd(Cuz,
                                                   [[j, n_] for (_, j) in In for n_ in [n, m]]),
                                      shape=[element_dim ** 2, 2])

                # cross cov between u_k in inds[m] and z_n z_m
                Cuk_znzm = tf.reshape(tf.gather_nd(Cuz,
                                                   [[k, n_] for (k, _) in Im for n_ in [n, m]]),
                                      shape=[element_dim ** 2, 2])

                # Marginal covar of zn and zm
                Cznzm = tf.reshape(tf.gather_nd(Czz, [*itertools.product([n, m], repeat=2)]),
                                   shape=[2, 2])
                # ... and its Cholesky decomposition.
                # ToDo: when n == m this is going to be rank 1. Handle
                #   it better, perhaps just cholesky by hand.
                Lzz = tf.linalg.cholesky(_add_diagonal_shift(Cznzm, 1e-6))

                # Covariance of the product u_j * u_k given z_n z_m
                Cujuk = tf.reshape(tf.gather_nd(Cuu, [[j, k] for (_, j) in In for (k, _) in Im]),
                                   shape=[element_dim ** 2, element_dim ** 2])
                Cujuk_znzm = (Cujuk
                              - Cuj_znzm @ tf.linalg.cholesky_solve(Lzz, tf.transpose(Cuk_znzm)))

                # and then this ugly thing...
                updates = tf.concat((
                    tf.gather_nd(znsq_exp_zn_plus_zm,
                                 [[n, m], [m, n]]),
                    tf.gather_nd(znzm_exp_zn_plus_zm,
                                 [[n, m], [m, n]])
                ), axis=-1)

                E_znzm_exp_zn_plus_zm = tf.scatter_nd(
                    [[0, 0], [1, 1], [0, 1], [1, 0]],
                    updates,
                    shape=[2, 2])

                val = tf.linalg.cholesky_solve(
                    Lzz, tf.transpose(tf.linalg.cholesky_solve(Lzz, E_znzm_exp_zn_plus_zm)))

                # gather the mean of z[n] and z[m]
                mean_z = tf.gather_nd(mz, [[n, ], [m, ]])

                uj_sub_uj_pred = (mean_uj[..., None]
                                  - Cuj_znzm @ tf.linalg.cholesky_solve(Lzz, mean_z[..., None]))
                uk_sub_uk_pred = (mean_uk[..., None]
                                  - Cuk_znzm @ tf.linalg.cholesky_solve(Lzz, mean_z[..., None]))

                # update values: Cov(uj, uk | zn, zm) * E[exp(zn + zm)]
                update_values = (Cujuk_znzm * tf.gather_nd(exp_zn_plus_zm, [[n, m]])
                                 + Cuj_znzm @ tf.matmul(val, Cuk_znzm, adjoint_b=True))
                update_values += (tf.matmul(uj_sub_uj_pred, uk_sub_uk_pred, adjoint_b=True)
                                  * tf.gather_nd(exp_zn_plus_zm, [[n, m]]))

                inv_covar_znzm_matmul_zn_exp_zn_plus_zm = tf.linalg.cholesky_solve(
                    Lzz,
                    tf.reshape(tf.gather_nd(zn_exp_zn_plus_zm, [[n, m], [m, n]]),
                               shape=[2, 1]))
                update_values += tf.matmul(uj_sub_uj_pred,
                                           tf.transpose(Cuk_znzm @ inv_covar_znzm_matmul_zn_exp_zn_plus_zm))
                update_values += tf.matmul(Cuj_znzm @ inv_covar_znzm_matmul_zn_exp_zn_plus_zm,
                                           uk_sub_uk_pred, adjoint_b=True)

                # Flatten the update_values and multiply them with the values from the
                # local stiffness matrix
                update_values = (tf.reshape(local_stiffness[n][:, None]
                                            * local_stiffness[m][None, :], [-1])
                                 * tf.reshape(update_values, [-1]))

                # Get the update inds = [(i, l) for i, _ in In for _, l in Im
                update_indices = [(i, l) for (i, _) in In for (_, l) in Im]

                EAuuTA += tf.scatter_nd(update_indices,
                                        update_values,
                                        [self.mesh.npoints, self.mesh.npoints])

        return EAuuTA


    def constraint_mc(self, mean, cov, f, nsamples):
        q = tfd.MultivariateNormalFullCovariance(mean, cov)
        samples = tf.squeeze(q.sample(nsamples))#, seed=1234))

        u_samples, z_samples = tf.split(samples, [self.mesh.npoints - self.mesh.nboundary_nodes,
                                                  self.mesh.n_elements],
                                        axis=-1)
        interior_inds = self.mesh.interior_node_indices()
        u_samples = tf.map_fn(lambda x: tf.scatter_nd(interior_inds[:, None],
                                                      x,
                                                      shape=[self.mesh.npoints]),
                              tf.squeeze(u_samples),
                              dtype=self.dtype)

        A_mc, _ = self.assemble(tf.exp(z_samples))
        A_mc = tf.linalg.LinearOperatorFullMatrix(A_mc)
        Au_mc = A_mc.matvec(u_samples)

        EAu = tf.reduce_mean(Au_mc, axis=0)
        EAu = tf.gather(EAu, self.mesh.interior_node_indices())

        c1 = EAu - f

        CovAu = tfp.stats.covariance(Au_mc, sample_axis=0)
        c2 = tf.linalg.cholesky(_add_diagonal_shift(CovAu, 1e-6))

        return c1, c2

    def constraint_mc_mf(self, u_dist, z_dist, f, nsamples):
        u_samples = tf.squeeze(u_dist.sample(nsamples))
        z_samples = tf.squeeze(z_dist.sample(nsamples))

        interior_inds = self.mesh.interior_node_indices()
        u_samples = tf.map_fn(lambda x: tf.scatter_nd(interior_inds[:, None],
                                                      x,
                                                      shape=[self.mesh.npoints]),
                              tf.squeeze(u_samples),
                              dtype=self.dtype)

        A_mc, _ = self.assemble(tf.exp(z_samples))
        A_mc = tf.linalg.LinearOperatorFullMatrix(A_mc)
        Au_mc = A_mc.matvec(u_samples)

        EAu = tf.reduce_mean(Au_mc, axis=0)
        EAu = tf.gather(EAu, self.mesh.interior_node_indices())

        c1 = EAu - f

        CovAu = tfp.stats.covariance(Au_mc, sample_axis=0)
        c2 = tf.linalg.cholesky(_add_diagonal_shift(CovAu, 1e-6))

        return c1, c2


    def fit(self,
            y,
            observation_noise_variance,
            log_diff_coeff_prior,
            penalty=1e-1,
            covar_structure='LowRankPlusDiagonal',
            lowrank_dim=10,
            u_mean_init=None,
            logdiff_coeff_mean_init=None,
            lowrank_factor_upper_init=None,
            lowrank_factor_lower_init=None,
            log_covar_diag_init=None,
            num_epochs=1000,
            D_chol_op=None,
            optimizer=None,
            multiplier=None,
            constraint_eval_method='MonteCarlo'):

        if multiplier is None:
            multiplier = tf.zeros([1], dtype=self.dtype)

        trainable_variables = _prepare_args(
            self,
            covar_structure,
            lowrank_dim,
            u_mean_init=u_mean_init,
            logdiff_coeff_mean_init=logdiff_coeff_mean_init,
            lowrank_factor_upper_init=lowrank_factor_upper_init,
            lowrank_factor_lower_init=lowrank_factor_lower_init,
            log_covar_diag_init=log_covar_diag_init)

        # things to be precomputed...
        obs_op = self.mesh.linear_interpolation_operator(self.index_points)
        obs_opt_obs_op = tf.matmul(obs_op, obs_op, adjoint_a=True)

        _, load_vector = self.assemble()
        interior_load_vector = tf.gather(load_vector, self.mesh.interior_node_indices())

        N = self.mesh.npoints - self.mesh.nboundary_nodes + self.mesh.n_elements
        s1 = 1.
        s2 = 1.

        def objective_function(arg, penalty):
            # unpack trainable variables
            (u_mean,
             logdiff_coeff_mean,
             log_covar_diag,
             lowrank_factor_upper,
             lowrank_factor_lower) = arg

            lowrank_factor = tf.concat(
                (tf.linalg.LinearOperatorLowerTriangular(lowrank_factor_upper).to_dense(),
                 lowrank_factor_lower), axis=-2)

            cov = (tf.matmul(lowrank_factor, lowrank_factor, adjoint_b=True)
                   + tf.linalg.diag(tf.exp(log_covar_diag)))

            # partition the complete covariance
            Cuu, Cuz, Czu, Czz = _partition_covar(cov,
                                                  self.mesh.npoints - self.mesh.nboundary_nodes,
                                                  self.mesh.n_elements)

            Lzz = tf.linalg.cholesky(Czz)

            # conditional covariance of u given z
            Cu_z = Cuu - Cuz @ tf.linalg.cholesky_solve(Lzz, Czu)

            ######################################
            #                                    #
            # E[-log p(y | u)] - E[ H(q(u|z)) ]  #
            #                                    #
            ######################################

            (u_mean_inflated,
             Cuu_inflated) = self.handle_zero_dirichlet_boundary_conditions(u_mean, Cuu)

            tr_oto_covu = tf.reduce_sum(obs_opt_obs_op * Cuu_inflated)
            z = y[..., None] - obs_op @ u_mean_inflated[:, None]
            expec_log_obs_pdf = -.5 * (tr_oto_covu
                                       + tf.reduce_sum(tf.matmul(z, z, adjoint_a=True))) / observation_noise_variance

            # this should have the conditioned mean -- but we only take the entropy
            # of it so the mean doesn't matter
            qu_z = tfd.MultivariateNormalFullCovariance(
                loc=u_mean, covariance_matrix=Cu_z)

            ######################
            #                    #
            # Dkl(q(z) || p(z) ) #
            #                    #
            ######################
            qz = tfd.MultivariateNormalTriL(
                loc=logdiff_coeff_mean, scale_tril=Lzz)

            dkl = qz.kl_divergence(log_diff_coeff_prior.get_marginal_distribution())

            mean = tf.concat([u_mean, logdiff_coeff_mean], axis=-1)

            if constraint_eval_method == 'MonteCarlo':
                c1, c2 = self.constraint_mc(mean, cov, interior_load_vector, 10)
            else:
                c1, c2 = self.constraint(mean, cov, load_vector)

            # apply the preconditioners
            if D_chol_op is not None:
                interior_inds = self.mesh.interior_node_indices()
                c1 = tf.scatter_nd(interior_inds[:, None],
                                   c1,
                                   shape=[self.mesh.npoints])

                c1 = tf.squeeze(D_chol_op.solvevec(c1))
                c2 = tf.squeeze(D_chol_op.solvevec(c2))

            c = tf.concat((c1, tf.reshape(c2, [-1])), axis=-1)

            return (-expec_log_obs_pdf + dkl - qu_z.entropy()
                    + .5*(s1*tf.reduce_sum(c1**2)
                          + s2*tf.reduce_sum(c2**2)
                          - tf.reduce_sum(c*multiplier)
                          )/penalty**2)

        @tf.function
        def grad(x):
            with tf.GradientTape() as tape:
                loss_value = objective_function(x, penalty)
            return loss_value, tape.gradient(loss_value, x)


        losses = []
        for epoch in range(num_epochs):
            loss_value, grads = grad(trainable_variables)
            optimizer.apply_gradients([(dx_, x_) for dx_, x_ in zip(grads, trainable_variables)])
            losses.append(loss_value)
            if epoch % 100 == 0:
                print("Epoch {} | Loss value {}".format(epoch, loss_value))


        import matplotlib.pyplot as plt
        plt.plot(losses)

        return trainable_variables

    def fit_mc(self,
               y,
               observation_noise_variance,
               log_diff_coeff_prior,
               penalty=1e-1,
               u_mean_init=None,
               logdiff_coeff_mean_init=None,
               u_scale_diag_init=None,
               logdiff_coeff_chol_init=None,
               num_epochs=1000,
               optimizer=None,
               constraint_eval_method='MonteCarloMF',
               D_chol_op=None):

        trainable_variables = _prepare_args_mf(
            self,
            u_mean_init=u_mean_init,
            logdiff_coeff_mean_init=logdiff_coeff_mean_init,
            u_scale_diag_init=u_scale_diag_init,
            logdiff_coeff_chol_init=logdiff_coeff_chol_init)

        # things to be precomputed...
        obs_op = self.mesh.linear_interpolation_operator(self.index_points)
        obs_opt_obs_op = tf.matmul(obs_op, obs_op, adjoint_a=True)

        _, load_vector = self.assemble()
        interior_load_vector = tf.gather(load_vector, self.mesh.interior_node_indices())

        def objective_function(arg, penalty):
            u_mean, logdiff_coeff_mean, u_diag_scale, logdiff_coeff_chol = arg

            Cuu = tf.linalg.diag(u_diag_scale**2)
            Cu_z = Cuu  # u is independent of z

            ######################################
            #                                    #
            # E[-log p(y | u)] - E[ H(q(u|z)) ]  #
            #                                    #
            ######################################

            (u_mean_inflated,
             Cuu_inflated) = self.handle_zero_dirichlet_boundary_conditions(u_mean, Cuu)

            tr_oto_covu = tf.reduce_sum(obs_opt_obs_op * Cuu_inflated)
            z = y[..., None] - obs_op @ u_mean_inflated[:, None]
            expec_log_obs_pdf = -.5 * (tr_oto_covu
                                       + tf.reduce_sum(tf.matmul(z, z, adjoint_a=True))) / observation_noise_variance

            # this should have the conditioned mean -- but we only take the entropy
            # of it so the mean doesn't matter
            qu_z = tfd.MultivariateNormalFullCovariance(
                loc=u_mean, covariance_matrix=Cu_z)

            ######################
            #                    #
            # Dkl(q(z) || p(z) ) #
            #                    #
            ######################
            qz = tfd.MultivariateNormalTriL(
                loc=logdiff_coeff_mean, scale_tril=logdiff_coeff_chol)

            dkl = qz.kl_divergence(log_diff_coeff_prior.get_marginal_distribution())

            ###############
            #             #
            # Constraints #
            #             #
            ###############
            if constraint_eval_method == 'MonteCarloMF':
                c1, c2 = self.constraint_mc_mf(
                    qu_z, qz, interior_load_vector, 10)

            return (-expec_log_obs_pdf + dkl - qu_z.entropy()
                    + .5 * (tf.reduce_sum(c1 ** 2) + tf.reduce_sum(c2 ** 2)) / penalty ** 2)

        @tf.function
        def grad(x):
            with tf.GradientTape() as tape:
                loss_value = objective_function(x, penalty)
            return loss_value, tape.gradient(loss_value, x)

        losses = []
        for epoch in range(num_epochs):
            loss_value, df = grad(trainable_variables)
            optimizer.apply_gradients([(dx_, x_)
                                       for dx_, x_ in zip(df, trainable_variables)])
            losses.append(loss_value)
            if epoch % 100 == 0:
                print("Epoch {} | Loss value {}".format(epoch, loss_value))

        import matplotlib.pyplot as plt
        plt.plot(losses)

        return trainable_variables

def _prepare_args(vbfem,
                  covar_structure,
                  lowrank_dim=10,
                  u_mean_init=None,
                  logdiff_coeff_mean_init=None,
                  lowrank_factor_upper_init=None,
                  lowrank_factor_lower_init=None,
                  log_covar_diag_init=None):
    # ToDo: Fix u mean to respect boundary nodes
    # useful dimensions
    nnodes = vbfem.mesh.npoints
    nboundarynodes = len(vbfem.mesh.boundary_nodes)
    nquadnodes = vbfem.mesh.n_elements

    D = nnodes + nquadnodes - nboundarynodes

    par_split = [nnodes - nboundarynodes,
                 nquadnodes,
                 D,
                 D * lowrank_dim]

    if u_mean_init is None:
        u_mean_init = tf.zeros(nnodes - nboundarynodes,
                               dtype=vbfem.dtype)

    if logdiff_coeff_mean_init is None:
        logdiff_coeff_mean_init = tf.zeros(nquadnodes,
                                           dtype=vbfem.dtype)

    # initialise U with iid normal
    if lowrank_factor_upper_init is None:
        lowrank_factor_upper_init = tf.linalg.diag(
            0.001*tf.ones(lowrank_dim, dtype=vbfem.dtype))

    if lowrank_factor_lower_init is None:
        lowrank_factor_lower_init = tf.random.normal(
            [D-lowrank_dim, lowrank_dim], stddev=0.001, dtype=vbfem.dtype)

    if log_covar_diag_init is None:
        log_covar_diag_init = tf.math.log(0.0001*tf.ones(D, dtype=vbfem.dtype))

    u_mean = tf.Variable(u_mean_init)
    logdiff_coeff_mean = tf.Variable(logdiff_coeff_mean_init)
    lowrank_factor_upper = tf.Variable(lowrank_factor_upper_init)
    lowrank_factor_lower = tf.Variable(lowrank_factor_lower_init)
    log_covar_diag = tf.Variable(log_covar_diag_init)

    return (u_mean,
            logdiff_coeff_mean,
            log_covar_diag,
            lowrank_factor_upper,
            lowrank_factor_lower)


def _prepare_args_mf(
        vbfem,
        u_mean_init=None,
        logdiff_coeff_mean_init=None,
        u_scale_diag_init=None,
        logdiff_coeff_chol_init=None):

    nnodes = vbfem.mesh.npoints
    nboundarynodes = vbfem.mesh.nboundary_nodes
    nquadnodes = vbfem.mesh.n_elements

    if u_mean_init is None:
        u_mean_init = tf.zeros(nnodes - nboundarynodes,
                               dtype=vbfem.dtype)

    if logdiff_coeff_mean_init is None:
        logdiff_coeff_mean_init = tf.zeros(nquadnodes, dtype=vbfem.dtype)

    if u_scale_diag_init is None:
        u_scale_diag_init = 1e-2*tf.ones(nnodes - nboundarynodes, dtype=vbfem.dtype)

    if logdiff_coeff_chol_init is None:
        logdiff_coeff_chol_init = tf.linalg.diag(
            1e-2*tf.ones(nquadnodes, dtype=vbfem.dtype))

    u_mean = tf.Variable(u_mean_init)
    u_scale_diag = tf.Variable(u_scale_diag_init)
    logdiff_coeff_mean = tf.Variable(logdiff_coeff_mean_init)
    logdiff_coeff_chol = tf.Variable(logdiff_coeff_chol_init)

    return u_mean, logdiff_coeff_mean, u_scale_diag, logdiff_coeff_chol

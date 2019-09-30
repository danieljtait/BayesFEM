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


class VariationalBayesLinearSecondOrderElliptic(LinearSecondOrderElliptic, VariationalBayesBaseFEM):

    def __init__(self,
                 coeff,
                 source,
                 mesh,
                 index_points=None,
                 name='VariationalBayesLinearSecondOrderElliptic'):
        """ Instantiate a VariationalBayesLinearSecondOrderElliptic object. """
        LinearSecondOrderElliptic.__init__(self, coeff, source, mesh)

        self._index_points = index_points

    def Eq_log_observation_density(self, y, mu, chol_u):
        """ Expected value of the log p(y|u) under Q(u).

        Parameters
        ----------

        y : Tensor
            Observations

        mu : Tensor
            Mean of Q(u)

        chol_u : Tensor
            Cholesky factor of the covariance of Q(u).

        Returns
        -------
        Elogprob : Tensor
            Expected value of lop p(y|u) under Q(u)

        ToDo: Rewrite this to take the already inflated versions
        """
        # conversion of cholesky to an operator
        cholu_op = tf.linalg.LinearOperatorLowerTriangular(chol_u)
        covu = cholu_op.matmul(cholu_op.to_dense(),
                               adjoint_arg=True)

        # (Possibly) inflate the parameters to
        # handle the boundary conditions
        muc, covuc = self.handle_zero_dirichlet_boundary_conditions(mu, covu)

        ####
        # Computation of :  Tr(O @ Suc @ Ot @ inv(Gamma))
        #

        # get the observation interpolation operator O
        O = self.mesh.linear_interpolation_operator(self.index_points)

        # get the scale operator from the observation noise model
        obs_scale_chol_op = self.obs_scale

        Ot_invGamma_O = tf.matmul(O, obs_scale_chol_op.solve(
            obs_scale_chol_op.solve(O), adjoint=True), adjoint_a=True)

        # Tr( Ot @ invGamma @ O @ covuc ) = vec(...).T @ vec( ... )
        tr_Ot_invGamma_O_covuc = tf.reduce_sum(Ot_invGamma_O * covuc)

        # (y - O @ muc).T GammaInv (y - O @ muc)
        #   = <z, GammaInv @ z>,  z = y - Omu
        z = y[..., None] - O @ muc[:, None]
        alpha = obs_scale_chol_op.solve(
            obs_scale_chol_op.solve(z), adjoint=True)

        zt_invGamma_z = tf.squeeze(tf.matmul(z, alpha, adjoint_a=True))

        return zt_invGamma_z + tr_Ot_invGamma_O_covuc


    def foo(self,
            mean_u, cov_u,
            mean_log_diff_coeff, cov_log_diff_coeff):

        mmT = mean_u[..., :, None] * mean_u[..., None, :] + cov_u

        EAmmTA = get_E_AQA(mean_log_diff_coeff,
                           cov_log_diff_coeff,
                           self,
                           mmT)

    def bar(self, y, muc, covuc, observation_noise_variance):
        with tf.name_scope("EObservationLogProb") as scope:
            # get the observation interpolation operator O
            obs_op = self.mesh.linear_interpolation_operator(self.index_points)

            obs_opT_obs_op = tf.matmul(obs_op, obs_op, adjoint_a=True)
            Tr_OtO_covu = tf.reduce_sum(obs_opT_obs_op * covuc)

            z = y[..., None] - obs_op @ muc[:, None]

            return -.5 * (Tr_OtO_covu
                          + tf.reduce_sum(tf.matmul(z, z, adjoint_a=True))) / observation_noise_variance

    def augmented_lagrangian(self,
                             y,
                             mean_u,
                             chol_u,
                             mean_log_diff_coeff,
                             chol_log_diff_coeff,
                             mean_multiplier,
                             covar_multiplier,
                             D):
        """

        Parameters
        ----------
        y : Tensor
            observed data

        y : Tensor
            Observed data

        mean_u : Tensor
            mean of u

        chol_u : Tensor
            cholesky factor of u

        mean_multiplier : Tensor
            Lagragian multiplier for the constraint on first moments

        covar_multiplier : Tensor
            Lagrangian multiplier for the constraint on second moments

        Returns
        -------

        """
        # E[-log p(y | u)] + DKL( q(theta) || p(theta) )
        #   - Tr( multi_1 @ chol(Au).T @ inv(D) ) - (multi_1 - f).T @ inv(D) (E[Au] - f)
        #   + (pen/2) * (Tr( chol(Au) @ chol(Au).T @ D )
        #               + (E[Au] - f).T @ inv(D) @ (E[Au] - f) )
        chol_D_op = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(D))

        invD = tf.linalg.inv(D)

        # Flow
        #   i)   inflate mean_u, chol_u
        #   ii)  turn mean_log_diff_coeff, and chol_log_diff_coeff into a distribution
        mean_u = boundary_util.inflate_interior_vector(mean_u, self.mesh)
        chol_u = boundary_util.inflate_interior_matrix(chol_u, self.mesh)

        #mean_multiplier = boundary_util.inflate_interior_vector(mean_multiplier, self.mesh)
        # -- pass covar_multiplier through cholesky so we ignore the upper triangle
        #covar_multiplier = boundary_util.inflate_interior_matrix(
        #    tf.linalg.LinearOperatorLowerTriangular(covar_multiplier).to_dense(),
        #    self.mesh)

        q_log_diff_coeff = tfp.distributions.MultivariateNormalTriL(
            loc=mean_log_diff_coeff,
            scale_tril=chol_log_diff_coeff
        )
        #   -------------------------------------------
        #   Components of the objective function
        #   -------------------------------------------
        #   ii)  compute E[log p(y | u)] with inflated mean_u, chol_u
        #   iii) compute Dkl( q(log_diff_coeff) || p(log_diff_coeff)

        # convert chol_u to a LinearOperator
        chol_u = tf.linalg.LinearOperatorLowerTriangular(chol_u)
        cov_u = chol_u.matmul(chol_u.to_dense(), adjoint=True)

        expect_logprob = self.bar(y, mean_u, cov_u, self.observation_noise_variance)
        dkl = q_log_diff_coeff.kl_divergence(self.log_diff_coeff_prior)

        objective_function = -expect_logprob + dkl
        #   -------------------------------------------
        #   Components of the constraint terms
        #   -------------------------------------------
        #   iii a) compute L := chol(cov(Au))
        #   iii b) compute Tr( L @ L.T @ inv(D))
        #   iv)  compute (E[Au] - b).T @ inv(D) @ (E[Au] - b)

        z, chol_op_covar_Au = self.constraint_vectors(
            mean_log_diff_coeff,
            q_log_diff_coeff.covariance(),
            mean_u,
            cov_u + tf.matmul(mean_u[:, None], mean_u[None, :]))

        # ToDo: must be a neater way of forming the covariance from choleksys
        #  -- actually this seems to be the standard way
        # From TensorflowProbabiltiy source code
        #   return self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)
        covar_Au = chol_op_covar_Au.matmul(
            chol_op_covar_Au.to_dense(), adjoint_arg=True)

        # z inv(D) z = chol_D.solve(z).H @ chol_D.solve(z)
        first_order_constraint = tf.reduce_sum(chol_D_op.solve(z) ** 2)

        second_order_constraint = tf.reduce_sum(covar_Au * invD)
        #   -------------------------------------------
        #   Components of the lagrange multiplier terms
        #   -------------------------------------------
        #   v)  compute Tr( multipiler_one @ L.T @ inv(D)
        #   vi) compute (multiplier_one @ inv(D) @ (E[Au] - b)
        #
        #   Return (objective function, constraint_one, constraint_two,
        with tf.name_scope("MultiplierCorrections") as scope:

            # Constraint term is given by c_i(x) =  [inv(chol(D)) @ (E[Au] - f) ]_i
            # so the Lagrange correction is
            #
            # multiplier.H @ c  =  multiplier.H @ chol_D.solve( E[Au] - f )
            #
            first_order_lagrange_correc = tf.squeeze(tf.matmul(
                mean_multiplier[:, None], chol_D_op.solve(z), adjoint_a=True))

            # This assumes that the multiplier corresponding to
            # the covariance constraint is upper triangular -- Fixed to Adjoint
            second_order_lagrange_correc = tf.reduce_sum(
                chol_op_covar_Au.matmul(covar_multiplier, adjoint_arg=True) * invD)

        return (objective_function,
                [first_order_constraint, second_order_constraint],
                [first_order_lagrange_correc, second_order_lagrange_correc],
                [z, chol_op_covar_Au.to_dense()])

    def constraint_vectors(self,
                    mean_log_diff_coeff,
                    cov_log_diff_coeff,
                    mean_u,
                    cov_u):
        """ Pair of constraints for augmented Lagrangian method.

        1. `E[A(z)u] - f`
        2. `chol_covar_Au` = chol(cov(Au))

        Parameters
        ----------

        mean_log_diff_coeff : Tensor
            Mean of the log diffusion coefficient.

        cov_log_diff_coeff : Tensor
            Covariance matrix of the log diffusion coefficient

        mean_u : Tensor
            Expected value of `u`.

        expected_u_outer : Tensor
            Expected value of outer(u, u).

        """
        with tf.name_scope("BuildConstraintVectors") as scope:

            expected_u_outer = mean_u[:, None] * mean_u[None, :] + cov_u

            # <variable names changed in constraint check>
            EAmmT, EAmmTA = bayesfem.vbfem.get_EAQA(
                mean_log_diff_coeff,
                cov_log_diff_coeff,
                self,
                expected_u_outer)


            EA, f = self.assemble(
                tf.exp(mean_log_diff_coeff
                       + .5 * tf.linalg.diag_part(cov_log_diff_coeff)))
            EAm = tf.matmul(EA, mean_u[:, None])

            # boundary correct EAmmTA
            bop = lambda x: bayesfem.vbfem.boundary_zero(x, self.mesh)
            ones_op = bayesfem.vbfem.boundary_ones_add_op(self.mesh)
            """
            # ToDo: X @ ones_op are likely to have more efficient implementations?
            #
            covar_Au = (bop(EAuuA)
                        + ones_op @ bop(EAuu)
                        + bop(EAuu) @ ones_op
                        + ones_op @ expected_u_outer @ ones_op)
            """

            """ Potential issue here... trying the above alternative implementation """
            EAmmTA = (bop(EAmmTA)
                      + ones_op @ bop(EAmmT)
                      + bop(EAmmT) @ ones_op
                      + ones_op @ expected_u_outer @ ones_op)

            covar_Au = EAmmTA - tf.matmul(EAm, EAm, adjoint_b=True)
            #covar_Au = EAmmTA - tf.matmul(f[:, None], f[None, :])

            # ToDo : Make sure adding ones doens't break anything here
            boundary_nodes = self.mesh.boundary_nodes
            covar_Au = tf.tensor_scatter_nd_add(
                covar_Au,
                [[i, i] for i in boundary_nodes],
                tf.ones(len(boundary_nodes), dtype=self.dtype)
            )

            # add a diag part\
            #jitter = 1e-4 * tf.ones([1], dtype=self.dtype)
            #covar_Au = _add_diagonal_shift(covar_Au, jitter)

            chol_op_covar_Au = tf.linalg.LinearOperatorLowerTriangular(
                tf.linalg.cholesky(covar_Au))

            return EAm - f[:, None], chol_op_covar_Au.to_dense()


    def first_order_auglagrangian(self,
                                  y,
                                  mean_u,
                                  chol_u,
                                  mean_log_diff_coeff,
                                  chol_log_diff_coeff,
                                  chol_D_op):
        """ Simpler version """

        mean_u = boundary_util.inflate_interior_vector(mean_u, self.mesh)
        chol_u = boundary_util.inflate_interior_matrix(chol_u, self.mesh)

        q_log_diff_coeff = tfp.distributions.MultivariateNormalTriL(
            loc=mean_log_diff_coeff,
            scale_tril=chol_log_diff_coeff
        )

        chol_u = tf.linalg.LinearOperatorLowerTriangular(chol_u)
        cov_u = chol_u.matmul(chol_u.to_dense(), adjoint=True)

        expect_logprob = self.bar(y, mean_u, cov_u, self.observation_noise_variance)
        dkl = q_log_diff_coeff.kl_divergence(self.log_diff_coeff_prior)

        objective_function = -expect_logprob + dkl

        ####
        # Construct the penalties

        # Evaluate the FEM at the mean
        cov_log_diff_coeff = q_log_diff_coeff.covariance()
        EA, f = self.assemble(
            tf.exp(mean_log_diff_coeff + .5 * tf.linalg.diag_part(cov_log_diff_coeff)))

        EA_sub_negf = EA @ mean_u[:, None] - f[:, None]

        constraints = chol_D_op.solve(EA_sub_negf)

        return objective_function, constraints

    def first_moment_constraint(self,
                                mean_u,
                                mean_log_diff_coeff,
                                cov_log_diff_coeff,
                                chol_D_op):
        """

        Returns chol(D).solve( E[Au] + f )

        Parameters
        ----------
        mean_u : Tensor
            Mean of marginal factor of `u`

        mean_log_diff_coeff : Tensor
            Mean of log diffusion coefficient.

        cov_log_diff_coeff : Tensor
            Covariance of log diffusion coefficient.

        chol_D_op : `tf.linalg.LinearOperatorLowerTriangular`
            Cholesky operator of the covariance operator D

        Returns
        -------

        first_moment_constraint : Tensor
            Mean constraint of the model.

        """
        expected_stiffness_matrix, f = self.assemble(
            tf.exp(mean_log_diff_coeff + .5 * tf.linalg.diag_part(cov_log_diff_coeff))
        )
        expected_stiffness_mat_sub_neg_f = expected_stiffness_matrix @ mean_u[:, None] - f[:, None]
        return chol_D_op.solve(expected_stiffness_mat_sub_neg_f)

    def second_moment_constraint(self,
                                 chol_D_op):
        """
        Returns chol_D.solve( chol( cov(Au) ) )

        Parameters
        ----------

        chol_D_op : `tf.linalg.LinearOperatorLowerTriangular`
            Cholesky operator of the covariance operator `D`.

        Returns
        -------
        """
        pass

    def fit_qu_factor(self,
                      y,
                      observation_noise_variance,
                      log_diff_coeff_dist,
                      D,
                      num_iterations):
        """ Uses the penalty method to learn the variational approximation to u.

        Parameters
        ----------

        y: Tensor
            Observations of the process.

        observation_noise_variance: Float
            Variance of the observations.

        log_diff_coeff_dist: `tfp.MultivariateNormal` distribution object.
            Fixed distribution of the log diffusion coefficient.

        D: Tensor
            Preconditioning precision matrix for the problem.

        num_iterations: integer
            No. of iterations of the outer problem.


        Returns
        -------

        u_dist: `tfp.MultivariateNormal` distribution object.
            Estimated variational distribution of u.
        """
        # take the cholesky of D and convert to an operator
        chol_D_op = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(D)
        )

        def _get_u_dist(mean_u, flat_scale_u):
            """ Helper function for assembling the distribution of u from parameters. """
            # inflate mean and scale
            interior_inds = self.mesh.interior_node_indices()

            mean_u = tf.scatter_nd(interior_inds[:, None],
                                   mean_u,
                                   shape=[self.mesh.npoints])

            flat_scale_u = tf.scatter_nd(interior_inds[:, None],
                                         flat_scale_u,
                                         shape=[self.mesh.npoints])

            return tfd.MultivariateNormalDiag(
                loc=mean_u,
                scale_diag=flat_scale_u,
                scale_identity_multiplier = 1e-6)

        # initial variables
        mean_u = tf.Variable(tf.ones(self.mesh.npoints - 2, dtype=self.dtype))
        scale_u = tf.Variable(0.1 * tf.ones(self.mesh.npoints - 2, dtype=self.dtype))

        # variable passed to objective function
        x = tf.Variable(tf.concat((mean_u, scale_u), axis=0))
        # split for unpacking the variable
        par_split = [self.mesh.npoints - 2, self.mesh.npoints - 2]

        # initial multiplier
        multiplier_size = (self.mesh.npoints
                           + self.mesh.npoints ** 2)
        multiplier = tf.Variable(tf.zeros(multiplier_size, dtype=self.dtype))

        # aug. lagrangian we will use to optimize
        auglagopt = bayesfem.auglagrangian.AugmentedLagrangianOptimizer(
            multiplier,
            tau=0.1,
            scale=tf.Variable([1.], dtype=self.dtype))


        @tf.function
        def objective_function(x):
            """ Objective function for the mean field factor of u."""
            (mean_u, flat_scale_u) = tf.split(x, par_split, axis=0)

            u_dist = _get_u_dist(mean_u, flat_scale_u)

            obs_op = self.mesh.linear_interpolation_operator(self.index_points)
            obs_opt_obs_op = tf.matmul(obs_op, obs_op, adjoint_a=True)

            covuc = u_dist.covariance()
            tr_oto_covu = tf.reduce_sum(obs_opt_obs_op * covuc)

            z = y[..., None] - obs_op @ u_dist.mean()[:, None]
            expec_log_obs_pdf = -.5 * (tr_oto_covu
                                       + tf.reduce_sum(tf.matmul(z, z, adjoint_a=True))) / observation_noise_variance

            # Entropy for the introduced approximation to q(u)
            qu_entropy = u_dist.entropy()

            return -expec_log_obs_pdf - qu_entropy

        @tf.function
        def constraint(x):
            (mean_u, flat_scale_u) = tf.split(x, par_split, axis=0)

            # inflate mean and scale
            interior_inds = self.mesh.interior_node_indices()

            mean_u = tf.scatter_nd(interior_inds[:, None],
                                   mean_u,
                                   shape=[self.mesh.npoints])

            flat_scale_u = tf.scatter_nd(interior_inds[:, None],
                                         flat_scale_u,
                                         shape=[self.mesh.npoints])

            u_dist = tfd.MultivariateNormalDiag(
                loc=mean_u,
                scale_diag=flat_scale_u,
                scale_identity_multiplier=1e-6)

            c1, c2 = self.constraint_vectors(log_diff_coeff_dist.mean(),
                                             log_diff_coeff_dist.covariance(),
                                             mean_u,
                                             tf.squeeze(u_dist.covariance()))

            c1 = tf.squeeze(chol_D_op.solve(c1))
            c2 = tf.reshape(chol_D_op.solve(c2), [-1])
            return tf.concat((c1, c2), axis=0)

        for nt in range(num_iterations):
            (x, consvals, _) = auglagopt.step_one(x,
                                                  objective_function,
                                                  constraint,
                                                  multiplier,
                                                  500,
                                                  method='Adams')

            constraints_norm_sq = tf.reduce_sum(consvals ** 2)

            if constraints_norm_sq <= auglagopt.constraint_tol + 2.:
                print("Adjusting multiplier")
                auglagopt.step_two(multiplier, consvals)

            else:
                auglagopt.step_three()
                print("... shrinking penalty to {}".format(auglagopt.penalty))

            cur_loss = objective_function(x)
            print("f = {}  |  || c ||^2 = {}".format(cur_loss, constraints_norm_sq))

        # inflate u and  return the distribution
        mean_u, scale_u = tf.split(x, par_split, axis=0)

        return _get_u_dist(mean_u, scale_u)

    def fit_qlog_diff_coeff_factor(self,
                                   u_dist,
                                   log_diff_coeff_prior_dist,
                                   D,
                                   mean_log_diff_coeff_init=None,
                                   chol_log_diff_coeff_init=None,
                                   num_iterations=5):
        """

        Parameters
        ----------

        Returns
        -------
        """

        # take the cholesky of D and convert to an operator
        chol_D_op = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(D)
        )

        if mean_log_diff_coeff_init is None:
            # initialize the variables
            mean_log_diff_coeff = tf.Variable(
                tf.zeros(self.mesh.n_elements, dtype=self.dtype))

            chol_log_diff_coeff = tf.Variable(
                tf.reshape(log_diff_coeff_prior_dist.get_marginal_distribution().scale.to_dense(), [-1]))


        else:
            mean_log_diff_coeff = mean_log_diff_coeff_init
            chol_log_diff_coeff = tf.reshape(chol_log_diff_coeff_init, [-1])

        # variable passed to objective function
        x = tf.Variable(tf.concat((mean_log_diff_coeff,
                                   chol_log_diff_coeff), axis=0))

        # split for unpacking the variable
        par_split = [self.mesh.n_elements,
                     self.mesh.n_elements**2]

        # initial multiplier
        multiplier_size = (0#self.mesh.npoints
                           + self.mesh.npoints ** 2)
        multiplier = tf.Variable(tf.zeros(multiplier_size, dtype=self.dtype))

        # aug. lagrangian we will use to optimize
        auglagopt = bayesfem.auglagrangian.AugmentedLagrangianOptimizer(
            multiplier,
            tau=0.1,
            scale=tf.Variable([1.], dtype=self.dtype))


        @tf.function
        def objective_function(x):
            (mean_log_diff_coeff,
             chol_log_diff_coeff) = tf.split(x, par_split, axis=0)

            chol_log_diff_coeff = tf.reshape(chol_log_diff_coeff,
                                             [self.mesh.n_elements,
                                              self.mesh.n_elements])

            log_diff_coeff_dist = tfd.MultivariateNormalTriL(
                loc=mean_log_diff_coeff,
                scale_tril=chol_log_diff_coeff)

            # KL divergence for the log diffusion coefficient
            dkl = tf.squeeze(
                log_diff_coeff_dist.kl_divergence(log_diff_coeff_prior_dist))

            # Monte-Carlo estimation of the log-det component
            thetas = log_diff_coeff_dist.sample(10, seed=1234)

            Az, f = self.assemble(tf.exp(thetas))
            inv_D_A = chol_D_op.solve(Az)

            _, log_abs_det = tf.linalg.slogdet(inv_D_A)
            log_abs_det = tf.reduce_mean(log_abs_det)

            return dkl + log_abs_det

        @tf.function
        def constraint(x):
            (mean_log_diff_coeff,
             chol_log_diff_coeff) = tf.split(x, par_split, axis=0)

            chol_log_diff_coeff = tf.reshape(chol_log_diff_coeff,
                                 [self.mesh.n_elements,
                                  self.mesh.n_elements])

            log_diff_coeff_dist = tfd.MultivariateNormalTriL(
                loc=mean_log_diff_coeff,
                scale_tril=chol_log_diff_coeff)

            c1, c2 = self.constraint_vectors(mean_log_diff_coeff,
                                              log_diff_coeff_dist.covariance(),
                                              tf.squeeze(u_dist.mean()),
                                              tf.squeeze(u_dist.covariance()))

            c1 = tf.squeeze(chol_D_op.solve(c1))
            c2 = tf.reshape(chol_D_op.solve(c2), [-1])
            return c2
            #return tf.concat((c1, c2), axis=0)

        for nt in range(num_iterations):
            (x, consvals,
             inner_prob_conv) = auglagopt.step_one(x,
                                                   objective_function,
                                                   constraint,
                                                   multiplier,
                                                   500,
                                                   method='Adams')
            constraints_norm_sq = tf.reduce_sum(consvals ** 2)

            if constraints_norm_sq <= auglagopt.constraint_tol: #+ 2.:
                print("Adjusting multiplier")
                auglagopt.step_two(multiplier, consvals)

            else:
                auglagopt.step_three()
                print("... shrinking penalty to {}".format(auglagopt.penalty))

            cur_loss = objective_function(x)
            print("f = {}  |  || c ||^2 = {}".format(cur_loss, constraints_norm_sq))

        (mean_log_diff_coeff,
         chol_log_diff_coeff) = tf.split(x, par_split, axis=0)

        chol_log_diff_coeff = tf.reshape(chol_log_diff_coeff,
                                         [self.mesh.n_elements,
                                          self.mesh.n_elements])

        return tfd.MultivariateNormalTriL(
                loc=mean_log_diff_coeff,
                scale_tril=chol_log_diff_coeff)


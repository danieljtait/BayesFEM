import tensorflow as tf


class AugmentedLagrangian:
    """
    Carries out minimisation of a constrained optimisation problem
    """
    def __init__(self,
                 objective_function,
                 constraints,
                 constraint_inner_products,
                 optimizer):

        with tf.name_scope("Init") as scope:
            self._objective_function = objective_function
            self._constraints = constraints
            self._constraint_inner_products = constraint_inner_products

            # store the optimizer
            self._optimizer=optimizer

    @property
    def objective_function(self):
        """ Objective function of the constrained problem"""
        return self._objective_function

    @property
    def constraints(self):
        """ list of constraints. """
        return self._constraints

    @property
    def constraint_inner_products(self):
        """ inner product inducing the norm of each penalty term. """
        return self._constraint_inner_products

    @property
    def optimizer(self):
        """ Optimizer used for the unconstrained problem. """
        return self._optimizer

    def get_lagrangian(self,
                       x,
                       lagrange_multipliers,
                       penalties):
        """
        Returns the augmented Lagrangian of the problem

        Parameters
        ----------
        x : list of tensors
            trainable variables for the model. List will
            be unpacked `self.objective_function(*x)`

        lagrange_multipliers : list of tensors
            Lagrange multipliers of the augmented Lagrangian form.

        penalties : list of tensors
            Penalty term for each constraint.

        Returns
        -------
        aug_lagrangian : tensor
            (batched) augmented Lagrangian

        """
        # get the value of the objective function
        objective_fun_term = self.objective_function(*x)

        # lagrangian multiplier correction
        lagr_multi_term = [-cip(lm, c(*x))
                           for lm, c, cip in zip(lagrange_multipliers,
                                                 self.constraints,
                                                 self.constraint_inner_prods)]

        # construct the correction for the penalty term
        penalty_term = [.5 * p * cip(c(*x), c(*x))
                        for p, c, cip in zip(penalties,
                                             self.constraints,
                                             self.constraint_inner_prods)]

        # probably going to break in graph mode?
        # ToDo: Make sure the sum() of tensors works in graph mode
        penalty_term = sum(penalty_term)

        return objective_fun_term + lagr_multi_term + penalty_term

    def fit(self,
            training_variables,
            lagrange_multipliers,
            penalties,
            penalty_steps,
            num_epochs=10,
            max_iter=100):
        """

        Parameters
        ----------
        training_variables: list of tensors
            Arguments of the objective function
            `loss = self.objective_function(*training_variables)`

        lagrange_multipliers: list of tensors
            Lagrange multipliers of the augmented Lagrangian form. Shape
            must agree with the corresponding constraint

        penalties: list of tensors
            Scalar penalty term corresponding to each constraint.

        penalty_steps: list of tensors
            Increments for the increasing penalty terms

        num_epochs : int
            Total number of updates of the penalty constraint.

        max_iter : int
            Maximum no. of iterations for each of the unconstrained subproblems.
        """

        # ToDo:
        #   Initialise Lagrangian multipliers and penalties, shape
        #   needs to agree with the output of the corresponding constraint

        def grad(x):
            with tf.GradientTape() as tape:
                loss = self.get_lagrangian(x,
                                             lagrange_multipliers,
                                             penalties)
            return loss, tape.gradient(loss, x)

        for epoch in range(num_epochs):
            nt = 0

            # optimize the unconstrained subproblem
            while nt < max_iter:
                loss_value, grads = grad(training_variables)
                self.optimizer.apply_gradients(
                    zip(grads, training_variables))
                # ToDo: Add break for the subproblem
                nt += 1

            # update the Lagrange multipliers and tensors
            for lm, p, c in zip(lagrange_multipliers,
                                penalties,
                                self.constraints):

                # update the Lagrange multiplier
                # lm_i = lm_i - p_i * c_i(x)
                v = c(*training_variables)
                lm.assign_sub(p * v)

                # update penalty
                p.assign_add(penalty_steps[i])

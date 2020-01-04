from typing import Callable, Union
import matplotlib.pyplot as plt

import numpy as np


def normal_pdf_unn(x, mean, variance, inv_variance=None):
    """
    Unnormalized pdf of a normal distribution.

    Parameters
    ----------
    x
        Where to evaluate the Normal.
    mean, variance:
        parameters of the gaussian.
    inv_variance
        Inverse of the variance (to avoid computing it again and again in certain cases).

    Returns
    -------
    Float: the result.
    """
    assert x.shape[0] == mean.shape[0] == variance.shape[0]
    assert x.ndim == mean.ndim
    if inv_variance is None:
        inv_variance = np.linalg.inv(variance)
    result = np.exp(-1 / 2 * (x - mean).T[np.newaxis, :] @ inv_variance @ (x - mean)[:, np.newaxis])
    assert result.size == 1
    return result


def log_normal_pdf_unn(x, mean, variance, inv_variance=None):
    """
    log of unnormalized pdf of a normal distribution.

    Parameters
    ----------
    x
        Where to evaluate the Normal.
    mean, variance:
        parameters of the gaussian.
    inv_variance
        Inverse of the variance (to avoid computing it again and again in certain cases).

    Returns
    -------
    Float: the result.
    """
    if inv_variance is None:
        inv_variance = np.linalg.inv(variance)
    return -1 / 2 * (x - mean).T[np.newaxis, :] @ inv_variance @ (x - mean)[:, np.newaxis]


class HastingMetropolis:

    def __init__(self, state: np.ndarray, pi: Callable[[np.ndarray], np.ndarray],
                 log_pi: Callable[[np.ndarray], np.ndarray] = None):
        self.dims = state.shape[0]
        self.state = state
        self.pi = pi
        self.log_pi = log_pi
        self.acceptance_rate = 0
        self.steps = 0
        self.history = {'state': [state], 'acceptance rate': []}

    def sample(self):
        """
        Draw one sample from the chain.

        Returns
        -------
        The new state.
        """
        self.steps += 1
        proposal = self.proposal_sampler()
        alpha = self.acceptance_ratio(proposal)
        u = np.random.uniform(0, 1)
        if u <= alpha:
            self.state = proposal
            self.acceptance_rate = ((self.steps - 1) * self.acceptance_rate + 1) / self.steps
        else:
            self.acceptance_rate = ((self.steps - 1) * self.acceptance_rate) / self.steps
        self.history['state'].append(self.state.copy())
        self.history['acceptance rate'].append(self.acceptance_rate)
        self.update_params(alpha=alpha)
        return self.state.copy()

    def multiple_sample(self, n: int):
        """
        Draw n samples from the chain.

        Parameters
        ----------
        n: # of samples.

        Returns
        -------
        An array of size (n, ndims).
        """
        result = np.empty((n, self.dims))
        for i in range(n):
            result[i] = self.sample()
        return result

    def update_params(self, *args, **kwargs):
        """
        For adaptative HM, you want to update the parameters of the proposal at each step. Fill this method to do so.
        """
        pass

    def proposal_sampler(self) -> np.ndarray:
        """
        Generate a proposal (function Q).
        """
        raise NotImplementedError

    def proposal_value(self, x, y):
        """
        Given x and y, returns the value of the proposal q(x, y).

        Parameters
        ----------
        x, y
            Values in X.

        Returns
        -------
        Float.
        """
        raise NotImplementedError

    def log_proposal_value(self, x, y):
        """
        Given x and y, returns the log of the proposal q(x, y).

        Parameters
        ----------
        x, y
            Values in X.

        Returns
        -------
        Float.
        """
        raise NotImplementedError

    def acceptance_ratio(self, proposal, log=True):
        """
        Compute the alpha parameter for the proposal, given the state we are in (self.state).

        Parameters
        ----------
        proposal
            The proposal value, given by e.g. proposal_sampler()
        log
            Computing acceptance_ratio using log trick

        Returns
        -------
        A float between 0 and 1.
        """
        if log and self.log_pi is not None:
            arg_exp = self.log_pi(proposal) - self.log_pi(self.state) \
                      + self.log_proposal_value(proposal, self.state) - self.log_proposal_value(self.state, proposal)
            alpha = np.exp(min(0, arg_exp))
        else:
            numerator = self.pi(proposal) * self.proposal_value(proposal, self.state)
            denominator = self.pi(self.state) * self.proposal_value(self.state, proposal)
            alpha = 1 if denominator == 0 else min(1, numerator / denominator)
            # Poor handling of overflow; best way would be to use log computation.
            # alpha = 0 if np.isnan(numerator) else alpha
            assert numerator >= 0 and denominator >= 0 or np.isnan(numerator), \
                "Numerator and denominator should be non-negative, got {} / {} at step {}.".format(
                    numerator, denominator, self.steps)
        assert 0 <= alpha <= 1, "Problem with the acceptance ratio. Expected a value between 0 and 1, got {}".format(
            alpha)
        return alpha

    def plot_acceptance_rates(self, offset=0):
        """
        Plot evolution of acceptance rate
        """
        plt.scatter(np.arange(offset, self.steps), self.history['acceptance rate'][offset:], s=2)

    def plot_autocorr(self, dim=0, maxlags=100, color='black', alpha=1, label=None):
        plt.acorr(np.array(self.history['state'])[:, dim], maxlags=maxlags, color=color, alpha=alpha, label=label)
        plt.xlabel('Lag')
        plt.xlim(-1, maxlags + 1)
        plt.ylabel('Autocorrelation')

    def mean_square_jump(self, stationarity: int):
        """
        Mean square jump of the chain.

        Parameters
        ----------
        stationarity: int
            Number of steps before reaching (numerical) stationarity.

        Returns
        -------
        float: (E(X_n - X_n-1)**2) ** 1/2
        """
        values = np.array(self.history['state'])[stationarity:, :]
        assert values.shape == (self.steps+1 - stationarity, self.dims)

        difference = (values[1:, :] - values[:-1, :])**2
        assert difference.shape == (self.steps - stationarity, self.dims)
        norm = np.sum(difference, axis=1)
        assert norm.shape == (self.steps - stationarity,)
        result = np.mean(norm, axis=0)
        return result


def truncated_drift(delta, grad_log_pi: Callable[[np.ndarray], np.ndarray]):
    """
    To get the truncated drift described in the article.

    Parameters
    ----------
    delta: float
        The delta parameter.
    grad_log_pi: Callable
        Function which returns the gradient of the log of the target distribution.

    Returns
    -------
    The truncated drift, ie. a function from R^d -> R^d.
    """

    def drift(x: np.ndarray):
        grad_log_pi_x = grad_log_pi(x)
        return delta * grad_log_pi_x / max(delta, np.linalg.norm(grad_log_pi_x))

    return drift


def projection_operators(epsilon_1, A_1):
    """
    Return the projection functions defined in the article.

    Parameters
    ----------
    epsilon_1, A_1:
        The two scaling operators.

    Returns
    -------
    proj_sigma: Projection on segment epsilon_1, A_1
    proj_gamma: Projection on cone of definite matrix of norm < 1_1
    proj_mu: Projection on centered ball of radius A_1
    """

    def proj_sigma(x: np.ndarray) -> np.ndarray:
        if x < epsilon_1:
            return epsilon_1
        elif x > A_1:
            return A_1
        else:
            return x

    def proj_gamma(x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2
        norm = np.linalg.norm(x, ord='fro')  # Frobenius norm
        if norm > A_1:
            print(f"Projection on Gamma! norm={norm:.1e}")
            return A_1 / norm * x
        else:
            return x

    def proj_mu(x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        norm = np.linalg.norm(x, ord=2)
        if norm > A_1:
            print(f"Projection on mu! norm={norm:.1e}")
            return A_1 / norm * x
        else:
            return x

    return proj_sigma, proj_gamma, proj_mu


def _check_values(epsilon_1, A_1, epsilon_2, tau_bar, mu: np.ndarray, gamma: np.ndarray,
                  threshold_start_estimate, threshold_use_estimate):
    """
    Check that the values comply with the article indication.
    """
    if not (0 < epsilon_1 < A_1 and 0 < epsilon_2):
        raise ValueError(f'epsilon_1, A_1 and epsilon_2 must verify: 0 < epsilon_1 < A_1, 0 < epsilon_2. Got:'
                         f'{epsilon_1:.1e}, {A_1:.1e}, {epsilon_2:.1e}')
    if not (0 < tau_bar < 1):
        raise ValueError(f'Target acceptation ratio should be between 0 and 1. Got: {tau_bar: .1e}.')
    if not (mu.ndim == 1 and gamma.ndim == 2):
        raise ValueError(f'Mu and gamma do not have the right dims. Expected 1 and 2, got {mu.ndim}, {gamma.ndim}')
    if not (mu.shape[0] == gamma.shape[0] == gamma.shape[1]):
        raise ValueError(f'mu and gamma should have the same shapes. Got: {mu.shape}, {gamma.shape}.')
    if not (threshold_start_estimate < threshold_use_estimate):
        raise ValueError(f'Start estimate should be below use estimate. '
                         f'Got: {threshold_start_estimate} > {threshold_use_estimate}')


class MALA(HastingMetropolis):

    def __init__(self, state, pi, log_pi,
                 drift: Callable[[np.ndarray], np.ndarray],
                 tau_bar: float,
                 gamma_0: np.ndarray,
                 sigma_0: float,
                 epsilon_2: float = 0,
                 ):
        """
        Basic MALA sampler.

        Parameters
        ----------
        state: initial state to start in.
        pi: Callable. Unnormalized pdf of the distribution we want to approximate.
        log_pi: log of the distribution we want to approximate
        drift: Callable.
        tau_bar: optimal acceptation rate.
        gamma_0, sigma_0: initial values for the parameters.
        epsilon_2: parameters of the HM algorithm. Must verify: 0 < epsilon_2.
        """
        super(MALA, self).__init__(state, pi, log_pi)

        self.drift = drift
        self.tau_bar = tau_bar
        self.gamma = gamma_0
        self.sigma = sigma_0
        self.epsilon_2 = epsilon_2

    def proposal_sampler(self) -> np.ndarray:
        big_lambda = self.gamma + self.epsilon_2 * np.eye(self.dims)
        mean = self.state + self.sigma ** 2 / 2 * big_lambda @ self.drift(self.state)
        variance = self.sigma ** 2 * big_lambda
        sample = np.random.multivariate_normal(mean=mean, cov=variance)
        return sample

    def proposal_value(self, x, y):
        assert x.shape == y.shape == self.state.shape
        big_lambda = self.gamma + self.epsilon_2 * np.eye(self.dims)
        mean = x + self.sigma ** 2 / 2 * big_lambda @ self.drift(x)
        variance = self.sigma ** 2 * big_lambda
        value = normal_pdf_unn(y, mean, variance)
        return value

    def log_proposal_value(self, x, y):
        assert x.shape == y.shape == self.state.shape
        big_lambda = self.gamma + self.epsilon_2 * np.eye(self.dims)
        mean = x + self.sigma ** 2 / 2 * big_lambda @ self.drift(x)
        variance = self.sigma ** 2 * big_lambda
        value = log_normal_pdf_unn(y, mean, variance)
        return value

    def plot_acceptance_rates(self, offset=0):
        super(MALA, self).plot_acceptance_rates(offset)
        plt.plot([offset - 1, self.steps + 1], [self.tau_bar, self.tau_bar], c='r', linestyle='--',
                 label=r'$\hat{\tau}$')



class AdaptiveMALA(MALA):
    def __init__(self, state, pi, log_pi,
                 drift: Callable[[np.ndarray], np.ndarray],
                 epsilon_1: float,
                 epsilon_2: float,
                 A_1: float,
                 tau_bar: float,
                 mu_0: np.ndarray,
                 gamma_0: np.ndarray,
                 sigma_0: float,
                 robbins_monroe=10,
                 threshold_start_estimate=1000,
                 threshold_use_estimate=1100,
                 ):
        """
                Adaptative MALA sampler, described in [1].

                Parameters
                ----------
                state: initial state to start in.
                pi: Callable. Unnormalized pdf of the distribution we want to approximate.
                log_pi: log of the distribution we want to approximate
                drift: Callable.
                epsilon_1, epsilon_2, A_1: parameters of the HM algorithm. Must verify: 0 < epsilon_1 < A_1, 0 < epsilon_2.
                tau_bar: target optimal acceptation rate.
                mu_0, gamma_0, sigma_0: initial values for the parameters.
                robbins_monroe: constant c_0 for the robbins monroe coefficients: g_n = c_0/n
                threshold_use_estimate: int corresponding to the number of steps after which we start updating the covariance
                matrix

                References
                ----------
                [1] An adaptive version for the Metropolis adjusted Langevin algorithm with a truncated drift, Yves F. Atchadé

                """
        super(AdaptiveMALA, self).__init__(state, pi, log_pi, drift, tau_bar, gamma_0, sigma_0, epsilon_2)
        _check_values(epsilon_1, A_1, epsilon_2, tau_bar, mu_0, gamma_0, threshold_start_estimate, threshold_use_estimate)
        self.epsilon_1 = epsilon_1

        self.A_1 = A_1
        self.mu = mu_0
        self.c_0 = robbins_monroe
        self.proj_sigma, self.proj_gamma, self.proj_mu = projection_operators(epsilon_1, A_1)
        self.threshold_use_estimate = threshold_use_estimate
        self.threshold_start_estimate = threshold_start_estimate
        self.params_history = {'mu': [mu_0.copy()],
                               'gamma': [gamma_0.copy()],
                               'sigma': [sigma_0]}
    def update_params(self, alpha):
        _update_params_adaptive(self, alpha)


class SymmetricRW(MALA):
    def __init__(self, state, pi, log_pi, gamma_0, sigma_0=1, epsilon_2=0):
        """
                A symmetric random walk HM sampler.

                Parameters
                ----------
                state: initial state.
                pi: distribution we want to approximate.
                log_pi: log of the distribution we want to approximate
                gamma_0: scale parameter for the proposal distribution.
                """
        super(SymmetricRW, self).__init__(state, pi, log_pi,
                                          drift=lambda x: np.zeros(x.shape),
                                          tau_bar=0.234,
                                          gamma_0=gamma_0,
                                          sigma_0=sigma_0,
                                          epsilon_2=epsilon_2)


class AdaptiveSymmetricRW(SymmetricRW):
    def __init__(self, state, pi, log_pi,
                 epsilon_1: float,
                 epsilon_2: float,
                 A_1: float,
                 tau_bar: float,
                 mu_0: np.ndarray,
                 gamma_0: np.ndarray,
                 sigma_0: float,
                 robbins_monroe=10,
                 threshold_start_estimate=1000,
                 threshold_use_estimate=1100
                 ):
        """
                An Adaptive symmetric random walk HM sampler.

                Parameters
                ----------
                state: initial state.
                pi: distribution we want to approximate.
                log_pi: log of the distribution we want to approximate
                gamma_0: scale parameter for the proposal distribution.
                threshold_start_estimate: int corresponding to the number of steps after which we start updating the
                covariance matrix
                """
        super(AdaptiveSymmetricRW, self).__init__(state, pi, log_pi,
                                                  gamma_0=gamma_0,
                                                  sigma_0=sigma_0,
                                                  epsilon_2=epsilon_2
                                                  )
        _check_values(epsilon_1, A_1, epsilon_2, tau_bar, mu_0, gamma_0, threshold_start_estimate, threshold_use_estimate)

        self.epsilon_1 = epsilon_1

        self.A_1 = A_1
        self.mu = mu_0
        self.c_0 = robbins_monroe
        self.proj_sigma, self.proj_gamma, self.proj_mu = projection_operators(epsilon_1, A_1)
        self.threshold_start_estimate = threshold_start_estimate
        self.threshold_use_estimate = threshold_use_estimate
        self.params_history = {'mu': [mu_0.copy()],
                               'gamma': [gamma_0.copy()],
                               'sigma': [sigma_0]}
    def update_params(self, alpha):
        _update_params_adaptive(self, alpha)


def _update_params_adaptive(model: Union[AdaptiveMALA, AdaptiveSymmetricRW], alpha: float):
    """
    Function to avoid duplicate between AdaptiveMALA and AdaptiveSymmetricRW.

    Parameters
    ----------
    model: instance of AdaptiveMALA or AdaptiveSymmetricRW
    alpha: the acceptance ratio

    Updates the parameters of the instance.
    """
    coeff = model.c_0 / model.steps

    model.mu = model.proj_mu(model.mu + coeff * (model.state - model.mu))
    # _gamma_estimate holds the estimation of the covariance matrix.
    # It is different from gamma: indeed, we want to be able to estimate the covariance matrix without using it
    # at first.
    if model.steps > model.threshold_start_estimate:
        covariance = (model.state - model.mu)[:, np.newaxis] @ (model.state - model.mu)[np.newaxis, :]
        model._gamma_estimate = model.proj_gamma(model.gamma + coeff * (covariance - model.gamma))
    if model.steps > model.threshold_use_estimate:
        model.gamma = model._gamma_estimate
    model.sigma = model.proj_sigma(model.sigma + coeff * (alpha - model.tau_bar))

    model.params_history['mu'].append(model.mu.copy())
    model.params_history['gamma'].append(model.gamma.copy())
    model.params_history['sigma'].append(model.sigma)


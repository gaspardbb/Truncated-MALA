import numpy as np

from hasting_metropolis import normal_pdf_unn, log_normal_pdf_unn, truncated_drift, MALA, AdaptiveMALA, SymmetricRW, \
    AdaptiveSymmetricRW


def banana(B, dim):
    """
    Returns three callables:
    * the unnormalized pdf of the banana distribution.
    * The log of the previous one
    * the gradient of the log of this pdf (useful for the drift).

    Parameters
    ----------
    B: float
        Size of the banana
    dim:
        # of dimensions. >1.

    Returns
    -------
    pdf, log_pdf, grad_log_pdf
    """
    def log_pdf(x):
        assert x.shape == (dim,)
        result = -x[0] ** 2 / 200 - 1 / 2 * (x[1] + B * x[0] ** 2 - 100 * B) ** 2 - 1 / 2 * np.sum(x[2:] ** 2)
        return result

    def pdf(x):
        assert x.shape == (dim,)
        result = np.exp(log_pdf(x))
        return result

    def grad_log_pdf(x):
        assert x.shape == (dim,)
        result = np.zeros(dim)
        result[0] = -x[0] / 100 - 2 * (x[1] + B * x[0] ** 2 - 100 * B)
        result[1] = -(x[1] + B * x[0] ** 2 - 100 * B)
        result[2:] = -x[2:]
        return result

    return pdf, log_pdf, grad_log_pdf


def product_of_gaussian(mus: np.ndarray, sigmas: np.ndarray):
    """
    Returns three callables:
    * the unnormalized pdf of a product of gaussian distribution.
    * The log of the previous one
    * the gradient of the log of this pdf (useful for the drift).

    Parameters
    ----------
    mus
        Means of the Gaussians. Shape: (n_gaussians, n_dims).
    sigmas
        Covariance of the Gaussians. Shape: (n_n_gaussians, n_dims, n_dims).

    Returns
    -------
    Float: value of the pdf.
    """
    assert mus.ndim == 2 and sigmas.ndim == 3
    assert mus.shape[0] == sigmas.shape[0] and mus.shape[1] == sigmas.shape[1] == sigmas.shape[2]
    dim = mus.shape[1]
    n_gaussians = mus.shape[0]
    inv_sigmas = np.zeros_like(sigmas)
    for j in range(n_gaussians):
        inv_sigmas[j] = np.linalg.inv(sigmas[j])

    def pdf(x):
        assert x.shape == (dim,)
        result = 1
        for i in range(n_gaussians):
            result *= normal_pdf_unn(x, mus[i], sigmas[i], inv_variance=inv_sigmas[i])
        return result

    def log_pdf(x):
        assert x.shape == (dim,)
        result = 0
        for i in range(n_gaussians):
            result += log_normal_pdf_unn(x, mus[i], sigmas[i], inv_variance=inv_sigmas[i])
        return result

    def grad_log_pdf(x):
        assert x.shape == (dim,)
        result = 0
        for i in range(n_gaussians):
            result += inv_sigmas[i] @ (x - mus[i])
        return result

    return pdf, log_pdf, grad_log_pdf


def random_product_of_gaussian(n_gaussians: int = 5):
    # Means between 0 and 1
    target_mus = np.random.uniform(0, 1, (n_gaussians, 2))
    # Isotropic
    target_sigmas = np.stack([np.eye(2)] * n_gaussians, axis=0)
    return product_of_gaussian(target_mus, target_sigmas)


def example_gaussian(mu, Sigma, N):
    """
    Kept for backward compatibility

    Parameters
    ----------
    mu:
        means of the gaussian
    Sigma:
        Covariance of the gaussian
    N:
        # of samples

    Returns
    -------
    Dict of models.
    """
    dim = Sigma.shape[0]

    initial_state = np.zeros(dim)

    target_pdf, log_target_pdf, target_grad_log_pdf = product_of_gaussian(mus=np.array([mu]), sigmas=np.array([Sigma]))

    # Parameter of the model
    delta = 1000
    epsilon_1 = 1e-7
    A_1 = 1e7
    epsilon_2 = 1e-6
    tau_bar = .574
    mu_0 = np.zeros(dim)
    gamma_0 = np.eye(dim)

    sigma_rw = 1e-1
    sigma_MALA = 1e-1
    sigma_opt_MALA = 1.3e-1
    sigma_opt_rw = 5.5e-1

    drift = truncated_drift(delta=delta, grad_log_pi=target_grad_log_pdf)

    mala_model = MALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift, tau_bar=tau_bar,
                      gamma_0=gamma_0, sigma_0=sigma_MALA)

    t_mala_model = AdaptiveMALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift,
                                epsilon_1=epsilon_1, epsilon_2=epsilon_2, A_1=A_1, tau_bar=tau_bar, mu_0=mu_0,
                                gamma_0=gamma_0, sigma_0=sigma_MALA)

    rw_model = SymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, gamma_0=gamma_0, sigma_0=sigma_rw)

    t_rw_model = AdaptiveSymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, epsilon_1=epsilon_1,
                                     epsilon_2=epsilon_2, A_1=A_1, tau_bar=tau_bar, mu_0=mu_0, gamma_0=gamma_0,
                                     sigma_0=sigma_rw)

    opt_rw_model = SymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, gamma_0=Sigma,
                               sigma_0=sigma_opt_rw)

    opt_mala_model = MALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift, tau_bar=tau_bar,
                          gamma_0=Sigma, sigma_0=sigma_opt_MALA)

    models = {'MALA': mala_model, 'T-MALA': t_mala_model, 'OPT-MALA': opt_mala_model, 'SRW': rw_model,
              'T-SRW': t_rw_model, 'OPT-SRW': opt_rw_model}

    for _, model in models.items():
        for _ in range(N):
            model.sample()

    return models
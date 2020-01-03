import numpy as np
from hasting_metropolis import AdaptiveMALA, truncated_drift, SymmetricRW, normal_pdf_unn
import matplotlib.pyplot as plt


def product_of_gaussian(mus: np.ndarray, sigmas: np.ndarray):
    """
    Returns two callables:
    * the unnormalized pdf of a product of gaussian distribution.
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
    n_gaussians = mus.shape[0]
    inv_sigmas = np.zeros_like(sigmas)
    for i in range(n_gaussians):
        inv_sigmas[i] = np.linalg.inv(sigmas[i])

    def pdf(x):
        result = 1
        for i in range(n_gaussians):
            result *= normal_pdf_unn(x, mus[i], sigmas[i], inv_variance=inv_sigmas[i])
        return result

    def grad_log_pdf(x):
        result = 0
        for i in range(n_gaussians):
            result += inv_sigmas[i] @ (x - mus[i])
        return result

    return pdf, grad_log_pdf


if __name__ == '__main__':
    """
    2D example of the truncated drift.
    """
    n_gaussians = 5

    # Means between 0 and 1
    target_mus = np.random.uniform(0, 1, (n_gaussians, 2))
    # Isotropic
    target_sigmas = np.stack([np.eye(2)] * n_gaussians, axis=0)
    # Start in the middle
    initial_state = np.zeros(2) + .5

    # Target functions
    target_pdf, target_grad_log_pdf = product_of_gaussian(mus=target_mus, sigmas=target_sigmas)

    # Parameter of the model
    delta = 100
    epsilon_1 = 1e-7
    A_1 = 1e7
    epsilon_2 = 1e-6
    tau_bar = .574
    mu_0 = np.zeros(2)
    gamma_0 = np.eye(2)
    sigma_0 = 1

    drift = truncated_drift(delta=delta, grad_log_pi=target_grad_log_pdf)

    t_mala_model = AdaptiveMALA(state=initial_state, pi=target_pdf, drift=drift,
                                epsilon_1=epsilon_1, epsilon_2=epsilon_2, A_1=A_1, tau_bar=tau_bar,
                                mu_0=mu_0, gamma_0=gamma_0, sigma_0=sigma_0)

    rw_model = SymmetricRW(state=initial_state, pi=target_pdf, scale=gamma_0)

    for i in range(50000):
        t_mala_model.sample()
        rw_model.sample()


    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    t_mala_model.plot_acceptance_rates()
    plt.legend()
    plt.subplot(2, 1, 2)
    rw_model.plot_acceptance_rates()

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    t_mala_model.plot_autocorr(dim=1)
    plt.legend()
    plt.subplot(2, 1, 2)
    rw_model.plot_autocorr(dim=1)
    plt.show()

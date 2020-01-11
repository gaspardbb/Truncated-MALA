import numpy as np

from hasting_metropolis import normal_pdf_unn, log_normal_pdf_unn


def banana(B, dim):
    """
    Returns three callables:
    * the unnormalized pdf of the banana distribution.
    * the log of the previous one
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
    assert mus.ndim == 2 and sigmas.ndim == 3, "Got mus.ndim={}, sigmas_ndim={}".format(mus.ndim, sigmas.ndim)
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

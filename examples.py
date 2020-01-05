import numpy as np
from hasting_metropolis import AdaptiveMALA, truncated_drift, SymmetricRW, normal_pdf_unn, log_normal_pdf_unn, MALA, \
    AdaptiveSymmetricRW
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
    for j in range(n_gaussians):
        inv_sigmas[j] = np.linalg.inv(sigmas[j])

    def pdf(x):
        result = 1
        for i in range(n_gaussians):
            result *= normal_pdf_unn(x, mus[i], sigmas[i], inv_variance=inv_sigmas[i])
        return result

    def log_pdf(x):
        result = 0
        for i in range(n_gaussians):
            result += log_normal_pdf_unn(x, mus[i], sigmas[i], inv_variance=inv_sigmas[i])
        return result

    def grad_log_pdf(x):
        result = 0
        for i in range(n_gaussians):
            result += inv_sigmas[i] @ (x - mus[i])
        return result

    return pdf, log_pdf, grad_log_pdf


def example_prod_gauss(N):
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
    target_pdf, log_target_pdf, target_grad_log_pdf = product_of_gaussian(mus=target_mus, sigmas=target_sigmas)

    # Parameter of the model
    delta = 1000
    epsilon_1 = 1e-7
    A_1 = 1e4
    epsilon_2 = 1e-6
    tau_bar = .574
    mu_0 = np.zeros(2)
    gamma_0 = np.eye(2)
    sigma_0 = 1

    drift = truncated_drift(delta=delta, grad_log_pi=target_grad_log_pdf)

    t_mala_model = AdaptiveMALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift,
                                epsilon_1=epsilon_1, epsilon_2=epsilon_2, A_1=A_1, tau_bar=tau_bar,
                                mu_0=mu_0, gamma_0=gamma_0, sigma_0=sigma_0)

    rw_model = SymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, gamma_0=gamma_0)

    for i in range(N):
        t_mala_model.sample()
        rw_model.sample()

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    t_mala_model.plot_acceptance_rates()
    plt.legend()
    plt.subplot(2, 1, 2)
    rw_model.plot_acceptance_rates()

    plt.figure(figsize=(8, 4))
    t_mala_model.plot_autocorr(dim=1, color='r', alpha=0.5, label='T-MALA')
    rw_model.plot_autocorr(dim=1, color='b', alpha=0.5, label='SRW')
    plt.legend()
    plt.show()


def compare_acceptance_rates(models):
    i = 1
    n = len(models.keys())
    figx, figy = 4 * ((n + 1) // 2), 8
    plt.figure(figsize=(figx, figy))
    for name, model in models.items():
        plt.subplot((n + 1) // 2, 2, i)
        model.plot_acceptance_rates()
        plt.legend()
        plt.title(name)
        i += 1


def compare_autocorr(models):
    i = 1
    n = len(models.keys())
    figx, figy = 4 * ((n + 1) // 2), 8
    plt.figure(figsize=(figx, figy))
    for name, model in models.items():
        plt.subplot((n + 1) // 2, 2, i)
        model.plot_autocorr(dim=1, label=name)
        plt.legend()
        i += 1


def compare_mean_square_jump(models: dict, stationarity: int, ax=None):
    """
    Compare the mean square jumps accross different models.

    Parameters
    ----------
    models: dict
        A dictionary of models
    stationarity: int
        Number of steps before stationarity (cf. HastingMetropolis)
    ax: plt.Axes
        A Matplotlib axes
    """
    result = {k: models[k].mean_square_jump(stationarity=stationarity) for k in models.keys()}

    if ax is None:
        _, ax = plt.subplots()
    # ax.bar(range(len(result)), list(result.values()), align='center')
    # ax.set_xticks(range(len(result)), list(result.keys()))
    ax.bar(*zip(*result.items()))
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Square Jump")
    return result


def compare_efficiency(models: dict, dim=0, n_iter=50, n_stationarity=10000,  ax=None):
    """
        Compare the efficiency (mu_dim) accross different models.

        Parameters
        ----------
        models: dict
            A dictionary of models
        dim: int
            Efficiency computed on mu_dim
        n_iter:
            number of chains used to estimate mu_dim
        n_stationarity: int
            lentgh of the chain from which we consider stationarity is reached
        ax: plt.Axes
            A Matplotlib axes
        """

    result = {}

    for name, model in models.items():
        mu_dim = []
        for _ in range(n_iter):
            model.initialize()
            for _ in range(n_stationarity):
                model.sample()
            mu_dim.append(model.state[dim])
        mu_dim = np.mean(mu_dim)
        result[name] = mu_dim


    if ax is None:
        _, ax = plt.subplots()
    # ax.bar(range(len(result)), list(result.values()), align='center')
    # ax.set_xticks(range(len(result)), list(result.keys()))
    ax.bar(*zip(*result.items()))
    ax.set_xlabel("Model")
    ax.set_ylabel("Estimation of mu")
    return result


def compare_models(models):
    compare_acceptance_rates(models)
    compare_autocorr(models)
    compare_mean_square_jump(models, stationarity=1000)
    compare_efficiency(models, dim=0, n_iter=2, n_stationarity=200)
    plt.show()


def example_gaussian(mu, Sigma, N):
    dim = Sigma.shape[0]

    initial_state = np.zeros(dim)

    target_pdf, log_target_pdf, target_grad_log_pdf = product_of_gaussian(mus=np.array([mu]),
                                                                          sigmas=np.array([Sigma]))

    # Parameter of the model
    delta = 1000
    epsilon_1 = 1e-7
    A_1 = 1e7
    epsilon_2 = 1e-6
    tau_bar = .574
    mu_0 = np.zeros(dim)
    gamma_0 = np.eye(dim)

    sigma_rw = 5e-2
    sigma_MALA = 5e-2
    sigma_opt_MALA = 1.3e-1
    sigma_opt_rw = 5.5e-1

    drift = truncated_drift(delta=delta, grad_log_pi=target_grad_log_pdf)

    mala_model = MALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift, tau_bar=tau_bar,
                      gamma_0=gamma_0, sigma_0=sigma_MALA)

    t_mala_model = AdaptiveMALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift,
                                epsilon_1=epsilon_1, epsilon_2=epsilon_2, A_1=A_1, tau_bar=tau_bar,
                                mu_0=mu_0, gamma_0=gamma_0, sigma_0=sigma_MALA)

    rw_model = SymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, gamma_0=gamma_0, sigma_0=sigma_rw)

    t_rw_model = AdaptiveSymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf,
                                     epsilon_1=epsilon_1, epsilon_2=epsilon_2, A_1=A_1, tau_bar=tau_bar,
                                     mu_0=mu_0, gamma_0=gamma_0, sigma_0=sigma_rw
                                     )

    opt_rw_model = SymmetricRW(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, gamma_0=Sigma,
                               sigma_0=sigma_opt_rw)

    opt_mala_model = MALA(state=initial_state, pi=target_pdf, log_pi=log_target_pdf, drift=drift, tau_bar=tau_bar,
                          gamma_0=Sigma, sigma_0=sigma_opt_MALA)

    models = {'MALA': mala_model,
              'T-MALA': t_mala_model,
              'OPT-MALA': opt_mala_model,
              'SRW': rw_model,
              'T-SRW': t_rw_model,
              'OPT-SRW': opt_rw_model}

    for _, model in models.items():
        for _ in range(N):
            model.sample()

    return models


def example_20D(N):
    # load covariance matrix
    import urllib.request
    target_url = "http://dept.stat.lsa.umich.edu/~yvesa/tmalaexcov.txt"
    data = urllib.request.urlopen(target_url)
    Sigma = []
    for line in data:
        Sigma.append(list(map(float, str.split(str(line)[2:-5]))))
    Sigma = np.array(Sigma)
    dim = Sigma.shape[0]
    return example_gaussian(np.zeros(dim), Sigma, N)


def example_vanilla_gauss(dim, N):
    return example_gaussian(np.zeros(dim), np.eye(dim), N)


if __name__ == '__main__':
    # example_prod_gauss(200)
    # models = example_20D(20000)
    models = example_vanilla_gauss(dim=2, N=5000)
    # s = np.random.random(size=(6, 6))
    # models = example_gaussian(np.zeros(6), s @ s.T, N=20000)
    compare_models(models)
    plt.show()

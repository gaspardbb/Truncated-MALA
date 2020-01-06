import numpy as np

from hasting_metropolis import AdaptiveMALA, truncated_drift, SymmetricRW, MALA, \
    AdaptiveSymmetricRW
import matplotlib.pyplot as plt

from plot_utils import grid_evaluation, animation_model_states
from sampler_utils import example_gaussian, random_product_of_gaussian, banana


def _update_dict(to_update_dict: dict, default_dict: dict):
    """
    Update a dictionary with the default values.
    """
    for param in default_dict:
        if param not in to_update_dict.keys():
            to_update_dict[param] = default_dict[param]
    return to_update_dict


def test_models(target_pdf, log_target_pdf, target_grad_log_pdf,
                N, initial_state=np.zeros(2),
                return_target=False,
                params_t_mala: dict = {},
                params_rw: dict = {},
                params_t_rw: dict = {},
                params_mala: dict = {}):
    """
    2D example of the truncated drift.

    Parameters
    ----------
    N: int
        # of samples
    return_target: bool
        Whether to return the target function
    params_t_mala: dict
        Parameters to override in the default parameters dictionary for MALA. See default_params_t_mala.
    params_rw
        Same for RW. See default_params_default_params_rw.

    Returns
    -------
    Dictionary of models, or Dictionary of models, function.
    """
    # Common params
    common_params = {"state": initial_state, "pi": target_pdf, "log_pi": log_target_pdf}

    # Default parameters of the models
    # T-MALA
    default_params_t_mala = {'delta': 1000, 'epsilon_1': 1e-7, 'A_1': 1e4, 'epsilon_2': 1e-6, 'tau_bar': .574,
                             'mu_0': np.zeros(2), 'gamma_0': np.eye(2), 'sigma_0': 1, 'threshold_start_estimate': 1000,
                             'threshold_use_estimate': 5000, 'robbins_monroe': 10, }
    params_t_mala = _update_dict(params_t_mala, default_params_t_mala)
    drift = truncated_drift(delta=params_t_mala['delta'], grad_log_pi=target_grad_log_pdf)

    # MALA
    default_params_mala = {'tau_bar': .574, 'gamma_0': np.eye(2), 'sigma_0': 1}
    params_mala = _update_dict(params_mala, default_params_mala)

    # RW
    default_params_rw = {'gamma_0': np.eye(2), 'sigma_0': 1, 'epsilon_2': 0}
    params_rw = _update_dict(params_rw, default_params_rw)

    # T-RW
    default_params_t_rw = default_params_t_mala
    default_params_t_rw.pop('delta')
    default_params_t_rw['sigma_0'] = 1e-1
    params_t_rw = _update_dict(params_t_rw, default_params_t_rw)

    params_t_mala.pop('delta')
    t_mala_model = AdaptiveMALA(**common_params, drift=drift, **params_t_mala)
    mala_model = MALA(**common_params, drift=drift, **params_mala)
    rw_model = SymmetricRW(**common_params, **params_rw)
    t_rw_model = AdaptiveSymmetricRW(**common_params, **params_t_rw)

    for i in range(N):
        t_mala_model.sample()
        mala_model.sample()
        rw_model.sample()
        t_rw_model.sample()

    models = {'T-MALA': t_mala_model, 'SRW': rw_model, 'T-SRW': t_rw_model, 'MALA': mala_model}

    if return_target:
        return models, (log_target_pdf, target_pdf)
    return models


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


def compare_autocorr(models, dim=0):
    i = 1
    n = len(models.keys())
    figx, figy = 4 * ((n + 1) // 2), 8
    plt.figure(figsize=(figx, figy))
    for name, model in models.items():
        plt.subplot((n + 1) // 2, 2, i)
        model.plot_autocorr(dim=dim, label=name)
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


def compare_efficiency(models: dict, dim=0, n_iter=50, n_stationarity=10000, axes=None):
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
    axes: plt.Axes
        A list of Matplotlib axes
    """

    result = {'mu_dims': {}, 'std_errors': {}, 'efficiencies': {}}

    for name, model in models.items():
        mu_dim = []
        for _ in range(n_iter):
            model.initialize()
            for _ in range(n_stationarity):
                model.sample()
            mu_dim.append(model.state[dim])
        result['mu_dims'][name] = mu_dim.copy()
        result['std_errors'][name] = np.std(mu_dim)
        if name == 'SRW':
            ref_eff = result['std_errors'][name]

    for name, model in models.items():
        result['efficiencies'][name] = result['std_errors'][name] / ref_eff

    if axes is None:
        _, axes = plt.subplots(nrows=2, figsize=(8, 8))
    ax = axes[0]
    labels, mu_dims = zip(*result['mu_dims'].items())
    ax.boxplot(x=list(mu_dims), labels=list(labels))
    ax.plot([0, len(labels) + 1], [0, 0], linestyle='--', c='g')
    ax.set_xlabel("Model")
    ax.set_ylabel("Estimation of mu{}".format(dim + 1))

    ax = axes[1]
    ax.bar(*zip(*result['efficiencies'].items()))
    ax.set_xlabel("Model")
    ax.set_ylabel("Comparison of efficiencies")
    return result


def compare_models(models, dim=0, n_iter=50):
    compare_acceptance_rates(models)
    compare_autocorr(models, dim=dim)
    compare_mean_square_jump(models, stationarity=1000)
    compare_efficiency(models, dim=dim, n_iter=n_iter, n_stationarity=10000)
    plt.show()


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
    # models = example_20D(100000)
    # models = example_vanilla_gauss(dim=2, N=50000)
    # s = np.random.random(size=(6, 6))
    # models = example_gaussian(np.zeros(6), s @ s.T, N=20000)
    # compare_models(models, dim=0, n_iter=50)
    # plt.show()

    # Product of Gaussian
    # pdf = random_product_of_gaussian()
    # x_range = y_range = (-3, 3)

    # Banana
    pdf = banana(0.05, dim=2)
    x_range = (-15, 15)
    y_range = (-10, 10)

    models, (log_target_pdf, target_pdf) = test_models(*pdf, N=250, return_target=True,
                                                       params_t_mala={'threshold_start_estimate': 0,
                                                                      'threshold_use_estimate': 20, 'robbins_monroe': 5,
                                                                      'sigma_0': 100})
    # Gaussian : use log pdf
    # result = grid_evaluation(log_target_pdf, 200, x_range, y_range)
    # Banana : use pdf
    result = grid_evaluation(target_pdf, 200, x_range, y_range)

    animation = animation_model_states(models, result, x_range + y_range,
                                       n_start=0,
                                       n_end=200)

    animation.save('basic_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])

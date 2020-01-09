import numpy as np

from hasting_metropolis import AdaptiveMALA, truncated_drift, SymmetricRW, MALA, \
    AdaptiveSymmetricRW
import matplotlib.pyplot as plt

from utils.plot_utils import grid_evaluation, animation_model_states
from utils.sampler_utils import example_gaussian, banana
from utils.model_comparison_utils import compare_models


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
                params_adapt_t_mala=None,
                params_rw=None,
                params_adapt_rw=None,
                params_t_mala=None):
    """
    2D example of the truncated drift.

    Parameters
    ----------
    :param N: int
        # of samples
    :param return_target: bool
        Whether to return the target function
    :param params_adapt_t_mala: dict
        Parameters to override in the default parameters dictionary for MALA. See default_params_adapt_t_mala.
    :param params_rw
        Same for RW. See default_params_default_params_rw.
    :param params_adapt_rw: dict
        Same for fully adaptive RW
    :param params_t_mala: dict
        Same for t-MALA

    Returns
    -------
    Dictionary of models, or Dictionary of models, function.
    """

    # Common params
    if params_adapt_t_mala is None:
        params_adapt_t_mala = {}
    if params_rw is None:
        params_rw = {}
    if params_t_mala is None:
        params_t_mala = {}
    if params_adapt_rw is None:
        params_adapt_rw = {}

    common_params = {"state": initial_state, "pi": target_pdf, "log_pi": log_target_pdf}

    # Default parameters of the models
    # adaptive T-MALA
    default_params_adapt_t_mala = {'delta': 1000, 'epsilon_1': 1e-7, 'A_1': 1e4, 'epsilon_2': 1e-6, 'tau_bar': .574,
                                   'mu_0': np.zeros(2), 'gamma_0': np.eye(2), 'sigma_0': 1,
                                   'threshold_start_estimate': 1000,
                                   'threshold_use_estimate': 5000, 'robbins_monroe': 10, }
    params_adapt_t_mala = _update_dict(params_adapt_t_mala, default_params_adapt_t_mala)
    drift = truncated_drift(delta=params_adapt_t_mala['delta'], grad_log_pi=target_grad_log_pdf)

    # T-MALA
    default_params_t_mala = {'tau_bar': .574, 'gamma_0': np.eye(2), 'sigma_0': 1}
    params_t_mala = _update_dict(params_t_mala, default_params_t_mala)

    # RW
    default_params_rw = {'gamma_0': np.eye(2), 'sigma_0': 1, 'epsilon_2': 0}
    params_rw = _update_dict(params_rw, default_params_rw)

    # Adaptive RW
    default_params_adapt_rw = default_params_adapt_t_mala.copy()
    default_params_adapt_rw.pop('delta')
    default_params_adapt_rw['sigma_0'] = 1e-1
    params_adapt_rw = _update_dict(params_adapt_rw, default_params_adapt_rw)

    params_adapt_t_mala.pop('delta')
    adapt_t_mala_model = AdaptiveMALA(**common_params, drift=drift, **params_adapt_t_mala)
    t_mala_model = MALA(**common_params, drift=drift, **params_t_mala)
    rw_model = SymmetricRW(**common_params, **params_rw)
    adapt_rw_model = AdaptiveSymmetricRW(**common_params, **params_adapt_rw)

    for i in range(N):
        adapt_t_mala_model.sample()
        t_mala_model.sample()
        rw_model.sample()
        adapt_rw_model.sample()

    models = {'SRW': rw_model, 'Adapt-SRW': adapt_rw_model, 'T-MALA': t_mala_model, 'Adapt-T-MALA': adapt_t_mala_model}

    if return_target:
        return models, (log_target_pdf, target_pdf)
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
    # models = example_20D(50000)
    models = example_vanilla_gauss(dim=2, N=5000)
    # s = np.random.random(size=(6, 6))
    # models = example_gaussian(np.zeros(6), s @ s.T, N=20000)
    compare_models(models, dim=0, n_iter=3)
    plt.show()

    # Product of Gaussian
    # pdf = random_product_of_gaussian()
    # x_range = y_range = (-3, 3)

    # Banana
    pdf = banana(0.05, dim=2)
    x_range = (-15, 15)
    y_range = (-10, 10)
    #
    models, (log_target_pdf, target_pdf) = test_models(*pdf, N=1000, return_target=True,
                                                       params_adapt_t_mala={'threshold_start_estimate': 0,
                                                                            'threshold_use_estimate': 20,
                                                                            'robbins_monroe': 5,
                                                                            'sigma_0': 10})
    # # # Gaussian : use log pdf
    result = grid_evaluation(log_target_pdf, 200, x_range, y_range)
    # # # Banana : use pdf
    result = grid_evaluation(target_pdf, 200, x_range, y_range)
    animation = animation_model_states(models, result, x_range + y_range,
                                       n_start=0,
                                       n_end=100)

    # animation.save('basic_animation.html', fps=1, extra_args=['-vcodec', 'libx264'])

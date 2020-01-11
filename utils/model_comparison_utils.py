import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

import numpy as np

from hasting_metropolis import AdaptiveMALA, MALA, AdaptiveSymmetricRW, SymmetricRW, truncated_drift
from utils.plot_utils import grid_evaluation, animation_model_states


def _update_dict(to_update_dict: dict, *default_dicts: dict):
    """
    Update a dictionary with the default values.
    """
    for default_dict in default_dicts:
        for param in default_dict:
            if param not in to_update_dict.keys():
                to_update_dict[param] = default_dict[param]
    return to_update_dict


def test_models(target_pdf, log_target_pdf, target_grad_log_pdf,
                N, initial_state, optimal=False,
                return_target=False,
                params_adapt_t_mala=None,
                params_rw=None,
                params_adapt_rw=None,
                params_t_mala=None,
                params_opt_t_mala=None,
                params_opt_rw=None):
    """
    2D example of the truncated drift.

    Parameters
    ----------
    :param N: int
        # of samples
    :param optimal: bool
        Whether to compare with optimal models or not
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
    :param params_opt_rw: dict
        Same for optimal RW
    :param params_opt_t_mala: dict
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

    dim = initial_state.size

    common_params = {"state": initial_state, "pi": target_pdf, "log_pi": log_target_pdf, 'epsilon_1': 1e-7, 'A_1': 1e4,
                     'robbins_monroe': 10}

    # Default parameters of the models
    # adaptive T-MALA
    default_params_adapt_t_mala = {'delta': 1000, 'epsilon_2': 1e-6, 'tau_bar': .574,
                                   'mu_0': np.zeros(dim), 'gamma_0': np.eye(dim), 'sigma_0': 1,
                                   'threshold_start_estimate': 1000,
                                   'threshold_use_estimate': 5000}

    params_adapt_t_mala = _update_dict(params_adapt_t_mala, default_params_adapt_t_mala, common_params)
    drift = truncated_drift(delta=params_adapt_t_mala['delta'], grad_log_pi=target_grad_log_pdf)
    params_adapt_t_mala.pop('delta')
    default_params_adapt_t_mala.pop('delta')

    # T-MALA
    default_params_t_mala = {'tau_bar': .574, 'gamma_0': np.eye(dim), 'sigma_0': 1}
    params_t_mala = _update_dict(params_t_mala, default_params_t_mala, common_params)

    # RW
    default_params_rw = {'gamma_0': np.eye(dim), 'sigma_0': 1, 'epsilon_2': 0, 'tau_bar': .234}
    params_rw = _update_dict(params_rw, default_params_rw, common_params)

    # Adaptive RW
    default_params_adapt_rw = default_params_adapt_t_mala.copy()
    params_adapt_rw = _update_dict(params_adapt_rw, default_params_adapt_rw, common_params)

    adapt_t_mala_model = AdaptiveMALA(drift=drift, **params_adapt_t_mala)
    t_mala_model = MALA(drift=drift, **params_t_mala)
    rw_model = SymmetricRW(**params_rw)
    adapt_rw_model = AdaptiveSymmetricRW(**params_adapt_rw)

    if optimal:
        assert params_opt_rw is not None and 'gamma_0' in params_opt_rw and 'sigma_0' in params_opt_rw
        params_opt_rw = _update_dict(params_opt_rw, params_rw)
        opt_rw_model = SymmetricRW(**params_opt_rw)

        assert params_opt_t_mala is not None and 'gamma_0' in params_opt_t_mala and 'sigma_0' in params_opt_t_mala
        params_opt_t_mala = _update_dict(params_opt_t_mala, params_t_mala)
        opt_t_mala_model = MALA(drift=drift, **params_opt_t_mala)

    for i in range(N):
        adapt_t_mala_model.sample()
        t_mala_model.sample()
        rw_model.sample()
        adapt_rw_model.sample()
        if optimal:
            opt_t_mala_model.sample()
            opt_rw_model.sample()

    models = {'SRW': rw_model, 'Adapt-SRW': adapt_rw_model, 'T-MALA': t_mala_model, 'Adapt-T-MALA': adapt_t_mala_model}

    if optimal:
        models['Opt-SRW'] = opt_rw_model
        models['Opt-T-MALA'] = opt_t_mala_model

    if return_target:
        return models, (log_target_pdf, target_pdf)
    return models


def compare_acceptance_rates(models: dict):
    """
    Compare the acceptance rates of models. Plot evolution of acceptance rates during simulations

    Parameters
    ----------
    models: dict
        A dictionary of models
    """
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


def compare_autocorr(models: dict, dim: int = 0):
    """
    Compare the autocorrelograms of the models to see the mix pace.

    Parameters
    ----------
    models: dict
        A dictionary of models
    dim: int
        Dimension of the data on which the correlations are considered
    """
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
            model.reinitialize()
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
    ax.set_ylabel("Efficiency")
    return result


def compare_models_stats(models, dim=0, n_iter=50):
    compare_acceptance_rates(models)
    compare_autocorr(models, dim=dim)
    compare_mean_square_jump(models, stationarity=1000)
    compare_efficiency(models, dim=dim, n_iter=n_iter, n_stationarity=10000)
    plt.show()


def compare_models_dynamics(target_pdf, log_target_pdf, target_grad_log_pdf, N, initial_state,
                            x_range, y_range, optimal=False,
                            n_start=0, n_end=100,
                            save=False, plot_covariance=True, filename='model_dynamics',
                            threshold_start_estimate=10, threshold_use_estimate=20, robbins_monroe=5,
                            sigma_0=1, fps=10):
    pdf = (target_pdf, log_target_pdf, target_grad_log_pdf)
    models, (log_target_pdf, target_pdf) = test_models(*pdf, N=N, initial_state=initial_state, return_target=True,
                                                       params_adapt_t_mala={
                                                           'threshold_start_estimate': threshold_start_estimate,
                                                           'threshold_use_estimate': threshold_use_estimate,
                                                           'robbins_monroe': robbins_monroe,
                                                           'sigma_0': sigma_0},
                                                       params_adapt_rw={
                                                           'threshold_start_estimate': threshold_start_estimate,
                                                           'threshold_use_estimate': threshold_use_estimate,
                                                           'robbins_monroe': robbins_monroe,
                                                           'sigma_0': sigma_0},
                                                       params_rw={'sigma_0': sigma_0,
                                                                  'robbins_monroe': robbins_monroe},
                                                       params_t_mala={'sigma_0': sigma_0,
                                                                      'robbins_monroe': robbins_monroe}
                                                       )

    result = grid_evaluation(target_pdf, 300, x_range, y_range)
    anim = animation_model_states(models, result, x_range + y_range,
                                  n_start=n_start,
                                  n_end=n_end, plot_covariance=plot_covariance)

    if save:
        # Set up formatting for the movie file
        print('Saving animation')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Gaspard Beugnot, Antoine Grosnit'), bitrate=1800,
                        extra_args=['-vcodec', 'libx264'])

        t = [time.time()]

        def progress_callback(i, n, t=t):
            print('Saving frame {} of {} in {:.2f}s'.format(i, n, time.time() - t[-1]))
            t[0] = time.time()

        anim.save('animations/' + filename + '.mp4', writer=writer, progress_callback=progress_callback)

    return models

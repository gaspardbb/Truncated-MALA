import matplotlib.pyplot as plt
import numpy as np


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


def compare_models(models, dim=0, n_iter=50):
    compare_acceptance_rates(models)
    compare_autocorr(models, dim=dim)
    compare_mean_square_jump(models, stationarity=1000)
    compare_efficiency(models, dim=dim, n_iter=n_iter, n_stationarity=10000)
    plt.show()

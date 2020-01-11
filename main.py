import matplotlib.pyplot as plt
import numpy as np

from utils.model_comparison_utils import compare_models_stats, compare_models_dynamics, test_models
from utils.sampler_utils import banana, random_product_of_gaussian, product_of_gaussian


def example_20d_gauss(N, n_iter, mu_dim=0):
    # load covariance matrix of a 20-dimensional gaussian distribution with highly correlated components
    import urllib.request
    target_url = "http://dept.stat.lsa.umich.edu/~yvesa/tmalaexcov.txt"
    data = urllib.request.urlopen(target_url)
    Sigma = []
    for line in data:
        Sigma.append(list(map(float, str.split(str(line)[2:-5]))))
    Sigma = np.array(Sigma)
    dim = Sigma.shape[0]
    mu = np.zeros(dim)

    pdf = product_of_gaussian(mus=np.array([mu]), sigmas=np.array([Sigma]))

    # scale of the models
    sigma_rw = 1e-1
    sigma_mala = 1e-1
    sigma_opt_mala = 1.3e-1
    sigma_opt_rw = 5.5e-1

    params_t_mala = {'sigma_0': sigma_mala, 'epsilon_1': 1e-7, 'A_1': 1e7}
    params_rw = {'sigma_0': sigma_rw, 'epsilon_1': 1e-7, 'A_1': 1e7}

    params_adapt_t_mala = {'delta': 1000, 'epsilon_1': 1e-7, 'A_1': 1e7, 'epsilon_2': 1e-6, 'sigma_0': sigma_mala,
                           'threshold_start_estimate': 1000,
                           'threshold_use_estimate': 5000}

    params_adapt_rw = {'epsilon_1': 1e-7, 'A_1': 1e7, 'epsilon_2': 1e-6, 'sigma_0': sigma_rw,
                       'threshold_start_estimate': 1000,
                       'threshold_use_estimate': 5000}

    params_opt_rw = {'sigma_0': sigma_opt_rw, 'gamma_0': Sigma}
    params_opt_t_mala = {'sigma_0': sigma_opt_mala, 'gamma_0': Sigma}

    models = test_models(*pdf, N=N, optimal=True, initial_state=np.zeros(dim),
                         return_target=False,
                         params_adapt_t_mala=params_adapt_t_mala,
                         params_rw=params_rw,
                         params_adapt_rw=params_adapt_rw,
                         params_t_mala=params_t_mala,
                         params_opt_t_mala=params_opt_t_mala,
                         params_opt_rw=params_opt_rw)

    compare_models_stats(models, dim=mu_dim, n_iter=n_iter)
    plt.show()


def dynamics(pdf, x_range, y_range, N=100, save=False, n_start=0, n_end=100, plot_covariance=True, fps=10,
             filename='model_dynamics', initial_state=None, threshold_start_estimate=10, threshold_use_estimate=20,
             robbins_monroe=5,
             sigma_0=10):
    if initial_state is None:
        initial_state = np.zeros(2)

    compare_models_dynamics(*pdf, N, initial_state=initial_state,
                            x_range=x_range, y_range=y_range, optimal=False,
                            n_start=n_start, n_end=n_end, filename=filename,
                            save=save, plot_covariance=plot_covariance,
                            threshold_start_estimate=threshold_start_estimate,
                            threshold_use_estimate=threshold_use_estimate, robbins_monroe=robbins_monroe,
                            sigma_0=sigma_0, fps=fps)


def example_banana_dynamics(N=100, save=False, n_start=0, n_end=100, dim=2, plot_covariance=True, fps=10):
    pdf = banana(0.05, dim=dim)
    x_range = (-20, 20)
    y_range = (-15, 15)
    threshold_start_estimate = 100
    threshold_use_estimate = 200
    robbins_monroe = 5
    sigma_0 = 10

    dynamics(pdf, x_range=x_range, y_range=y_range, N=N, save=save, n_start=n_start, n_end=n_end,
             plot_covariance=plot_covariance, fps=fps, filename='banana',
             threshold_start_estimate=threshold_start_estimate,
             threshold_use_estimate=threshold_use_estimate, robbins_monroe=robbins_monroe, sigma_0=sigma_0)


def example_gaussian_dynamics(N=100, n_gaussians=5, save=False, n_start=0, n_end=100, plot_covariance=True, fps=10):
    pdf = random_product_of_gaussian(n_gaussians)
    x_range = y_range = (-2.5, 3.5)
    threshold_start_estimate = 100
    threshold_use_estimate = 200
    robbins_monroe = 5
    sigma_0 = 1

    dynamics(pdf=pdf, x_range=x_range, y_range=y_range, N=N, save=save, n_start=n_start, n_end=n_end,
             plot_covariance=plot_covariance, fps=fps, filename='gaussian',
             threshold_start_estimate=threshold_start_estimate,
             threshold_use_estimate=threshold_use_estimate, robbins_monroe=robbins_monroe, sigma_0=sigma_0)


if __name__ == '__main__':
    example_20d_gauss(5000, 50)
    example_gaussian_dynamics(500, save=True, n_start=0, n_end=500, fps=40)
    example_banana_dynamics(500, n_start=0, n_end=500, save=True, fps=40)

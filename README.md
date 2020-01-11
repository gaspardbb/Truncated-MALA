# Adaptive Metropolis Adjusted Langevin Algorithm with a truncated drift
Implementation of the Truncated-MALA HM sampler<sup>1</sup> (T-MALA) for the course of [Computational Statistics](https://sites.google.com/site/stephanieallassonniere/enseignements/methodes-mcmc-et-applications) gyven by Stéphanie Allassonière for [Master MVA](https://www.master-mva.com).

This sampler belongs to the family of [Hastings-Metropolis](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) algorithms, and more precisely to the class of [Metropolis-adjusted Langevin algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) (MALA) samplers which are [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) methods for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult.

This repository notably contains implementation of the following samplers ([hasting_metropolis.py](./hasting_metropolis.py)):
- Symmetric Random Walk (with normal proposal distribution whose scale parameter $\sigma$ is adaptive)
- Fully adaptive Symmetric Random Walk (with normal proposal distribution whose scale parameter $\sigma$ _and_ covariance matrix $\Gamma$ are adaptive)
- Fully adaptive T-MALA (implementation of [1])
- T-MALA (same as above except only the scale parameter $\sigma$ is adaptive)

The following are suggested as target distributions ([sampler_utils.py](/utils/sampler_utils.py)):
- Multivariate gaussian
- "Banana shape" distribution, whose density is given by:
$$\pi(x) &\propto e^{-\frac{x_1^2}{200} - \frac{1}{2}(x_2 + B x_1^2 - 100 B)^2- \frac{1}{2}(x_3^2 + \dots + x_d^2)}$$

In
### 
![](animations/gaussian.gif)
![](animations/banana.gif)


1. [An adaptive version for the Metropolis adjusted Langevin algorithm with a truncated drift](http://dept.stat.lsa.umich.edu/~yvesa/atmala.pdf), _Yves F. Atchadé, Methodology and Computing in Applied Probability, 2006_.

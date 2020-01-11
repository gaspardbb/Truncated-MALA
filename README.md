# Adaptive Metropolis Adjusted Langevin Algorithm with a truncated drift
Implementation of the Truncated-MALA HM sampler<sup>1</sup> (T-MALA) for the course of [Computational Statistics](https://sites.google.com/site/stephanieallassonniere/enseignements/methodes-mcmc-et-applications) gyven by Stéphanie Allassonière for [Master MVA](https://www.master-mva.com).

### Presentation
This sampler belongs to the family of [Hastings-Metropolis](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) algorithms, and more precisely to the class of [Metropolis-adjusted Langevin algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) (MALA) samplers which are [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) methods for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult.

### Samplers
This repository notably contains implementation of the following samplers ([hasting_metropolis.py](./hasting_metropolis.py)):
- Symmetric Random Walk (with normal proposal distribution whose scale parameter is adaptive)
- Fully adaptive Symmetric Random Walk (with normal proposal distribution whose scale parameter _and_ covariance matrix are adaptive)
- Fully adaptive T-MALA (implementation of [1])
- T-MALA (same as above except only the scale parameter is adaptive)

### Target distributions
The following are suggested as target distributions ([sampler_utils.py](/utils/sampler_utils.py)):
- Multivariate gaussian
- "Banana shape" distribution, whose density is given by:
![equation](https://latex.codecogs.com/gif.latex?\pi(x)&space;&\propto&space;\text{exp}(x_1^2&space;/&space;200&space;-&space;0.5&space;(x_2&space;&plus;&space;B&space;x_1^2&space;-&space;100&space;B)^2-&space;0.5&space;(x_3^2&space;&plus;&space;\dots&space;&plus;&space;x_d^2))
)

### Sampling examples
- Gaussian target distribution:
<p align="center">
  <img src="animations/gaussian.gif">
</p>

- "Banana shape" target distribution:
<p align="center">
  <img src="animations/banana.gif">
</p>


1. [An adaptive version for the Metropolis adjusted Langevin algorithm with a truncated drift](http://dept.stat.lsa.umich.edu/~yvesa/atmala.pdf), _Yves F. Atchadé, Methodology and Computing in Applied Probability, 2006_.

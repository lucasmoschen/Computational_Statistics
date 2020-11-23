# Computational Statistics (["Estatística Computacional"](https://emap.fgv.br/disciplina/doutorado/estatistica-computacional))

Course materials for Computational Statistics, a PhD-level course at [EMAp](http://emap.fgv.br/).

## Lecture notes and other resources

- We will be using the excellent [materials](http://www.stats.ox.ac.uk/~rebeschi/teaching/AdvSim/18/index.html) from Professor Patrick Rebeschini (Oxford University) as a general guide for our course. 

As complementary material,

- These lecture [notes](https://statweb.stanford.edu/~susan/courses/s227/)  by stellar statistician [Susan Holmes](https://statweb.stanford.edu/~susan/) are also well worth taking a look.

- [Monte Carlo theory, methods and examples](https://statweb.stanford.edu/~owen/mc/) by [Professor Art Owen](https://statweb.stanford.edu/~owen/), gives a nice and complete treatment of all the topics on simulation, including a whole chapter on variance reduction. 

Other materials, including lecture notes and slides may be posted here as the course progresses. 

[Here](https://github.com/maxbiostat/Computational_Statistics/blob/master/annotated_bibliography.md) you can find a nascent annotated bibliography with landmark papers in the field. 

## Books

Books marked with [a] are advanced material.

**Main**
- Gamerman, D., & Lopes, H. F. (2006). [Markov chain Monte Carlo: stochastic simulation for Bayesian inference](http://www.dme.ufrj.br/mcmc/). Chapman and Hall/CRC.
- Robert, C. P., Casella, G. (2004). [Monte Carlo Statistical Methods](https://www.researchgate.net/profile/Christian_Robert2/publication/2681158_Monte_Carlo_Statistical_Methods/links/00b49535ccaf6ccc8f000000/Monte-Carlo-Statistical-Methods.pdf). John Wiley & Sons, Ltd.

**Supplementary**
- Givens, G. H., & Hoeting, J. A. (2012). [Computational Statistics](https://www.stat.colostate.edu/computationalstatistics/) (Vol. 710). John Wiley & Sons.
- [a] Meyn, S. P., & Tweedie, R. L. (2012). [Markov chains and stochastic stability](https://www.springer.com/gp/book/9781447132691). Springer Science & Business Media.
- [a] Nummelin, E. (2004). [General irreducible Markov chains and non-negative operators](https://www.cambridge.org/core/books/general-irreducible-markov-chains-and-nonnegative-operators/0557D49C011AA90B761FC854D5C14983) (Vol. 83). Cambridge University Press.



## Interlude: Bayesian Statistics

Reference books are 

- Bernardo, J. M., & Smith, A. F. (2009). [Bayesian Theory](https://statisticalsupportandresearch.files.wordpress.com/2019/03/josc3a9-m.-bernardo-adrian-f.-m.-smith-bayesian-theory-wiley-1994.pdf)  (Vol. 405). John Wiley & Sons. (PS: link is to the 1994 edition).
- Robert, C. (2007). [The Bayesian Choice](https://errorstatistics.files.wordpress.com/2016/03/robert-20071.pdf). Springer-Verlag.
- Jaynes, E. T. (2003). Probability theory: The logic of science. Cambridge university press.

Some other material I mentioned during class:

- [This](https://normaldeviate.wordpress.com/2012/11/17/what-is-bayesianfrequentist-inference/) is the [Larry Wasserman](http://www.stat.cmu.edu/~larry/) blog post I discussed. 
- The vignette I mentioned is [here](https://cran.r-project.org/web/packages/LaplacesDemon/vignettes/BayesianInference.pdf).


## Simulation

- [Random Number Generation](https://www.iro.umontreal.ca/~lecuyer/myftp/papers/handstat.pdf) by [Pierre L'Ecuyer](http://www-labs.iro.umontreal.ca/~lecuyer/);
- [Non-Uniform Random Variate Generation](http://www.nrbook.com/devroye/) by the great [Luc Devroye](http://luc.devroye.org/);
- [Rejection Control and Sequential importance sampling](http://stat.rutgers.edu/home/rongchen/publications/98JASA_rejection-control.pdf) (1998), by Liu et al. discusses how to improve importance sampling by controlling rejections.

### Markov chain Monte Carlo

- Charlie Geyer's [website](http://users.stat.umn.edu/~geyer/) is a treasure trove of material on Statistics in general, MCMC methods in particular. 
See, for instance, [On the Bogosity of MCMC Diagnostics](http://users.stat.umn.edu/~geyer/mcmc/diag.html). 


## Optmisation
#### The EM algortithm 
- This elementary [tutorial](https://zhwa.github.io/tutorial-of-em-algorithm.html)  is simple but effective.
- The book [The EM algorithm and Extensions](https://books.google.com.br/books?hl=en&lr=&id=NBawzaWoWa8C&oi=fnd&pg=PR3&dq=The+EM+algorithm+and+Extensions&ots=tp68LOYAvP&sig=iCEMt5YUIMToTSESxLctWcob8VM#v=onepage&q=The%20EM%20algorithm%20and%20Extensions&f=false) is a well-cited resource.
- [Monte Carlo EM](https://github.com/bob-carpenter/case-studies/blob/master/monte-carlo-em/mcem.pdf) by Bob Carpenter (Columbia).

## Miscellanea

- In [these](https://terrytao.wordpress.com/2010/01/03/254a-notes-1-concentration-of-measure/) notes, [Terence Tao](https://en.wikipedia.org/wiki/Terence_Tao) gives insights into **concentration of measure**, which is the reason why integrating with respect to a probability measure in high-dimensional spaces is _hard_. 

- [A Primer for the Monte Carlo Method](https://archive.org/details/APrimerForTheMonteCarloMethod), by the great [Ilya Sobol](https://en.wikipedia.org/wiki/Ilya_M._Sobol), is one of the first texts on the Monte Carlo method.

- The Harris inequality, `E[fg] >= E[f]E[g]`, for `f` and `g` increasing, is a special case of the [FKG inequality](https://en.wikipedia.org/wiki/FKG_inequality). 

- In [Markov Chain Monte Carlo Maximum Likelihood](https://www.stat.umn.edu/geyer/f05/8931/c.pdf), Charlie Geyer shows how one can use MCMC to do maximum likelihood estimation when the likelihood cannot be written in closed-form.
This paper is an example of MCMC methods being used outside of Bayesian statistics.

### Extra (fun) resources

In these blogs and websites you will often find interesting discussions on computational, numerical and statistical aspects of applied Statistics and Mathematics.

- Christian Robert's [blog](https://xianblog.wordpress.com/);
- John Cook's [website](https://www.johndcook.com/blog/);
- [Statisfaction](https://statisfaction.wordpress.com/) blog.

# Algorithmic Recourse

This repository contains all the work produced in relation to my first research topic: the effect of endogenous domain and model shifts on algorithmic recourse.

## Environment

To facilitate reproducibility this repository works with its own environment. When you clone this repo, all its dependencies are already declared in 'Project.toml', so you should not have to install any missing packages manually. 

### Compatibility

In this project I use Turing.jl for the purpose of Bayesian Deep Learning. Unfortunately, Turing is currently not compatible with the latest version of Julia (1.7).  When using the code contained in this project you are therefore best advised to work in versions `julia = "1.3, 1.4, 1.5, 1.6"` as specified in 'Project.toml'.

### AlgorithmicRecourse.jl

There is a companion Julia package to this repository. It is not yet registered, but can be found on [Github](https://github.com/pat-alt/AlgorithmicRecourse.jl). As per above, you should not have to manually installed the package yourself as long as your working within the project environment. But should you anyway want to install the package to use in a different environment you can do that like so:

```julia
using Pkg
Pkg.add("https://github.com/pat-alt/AlgorithmicRecourse.jl")
```
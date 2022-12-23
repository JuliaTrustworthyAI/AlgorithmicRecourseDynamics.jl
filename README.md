
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/AlgorithmicRecourseDynamics.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/AlgorithmicRecourseDynamics.jl/dev) [![Build Status](https://github.com/pat-alt/AlgorithmicRecourseDynamics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/AlgorithmicRecourseDynamics.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/AlgorithmicRecourseDynamics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/AlgorithmicRecourseDynamics.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac) [![Twitter Badge](https://img.shields.io/twitter/url/https/twitter.com/paltmey.svg?style=social&label=Follow%20%40paltmey)](https://twitter.com/paltmey)

# AlgorithmicRecourseDynamics

`AlgorithmicRecourseDynamics.jl` is a small package for modeling Algorithmic Recourse Dynamics. It builds on `CounterfactualExplanations`, a package for generating counterfactual explanations.

## Basic Usage

### Data and Model

``` julia
N = 1000
xmax = 2
X, ys = make_blobs(
    N, 2; 
    centers=2, as_table=false, center_box=(-xmax => xmax), cluster_std=0.1
)
ys .= ys.==2
X = X'
counterfactual_data = CounterfactualData(X,ys')
```

``` julia
n_epochs = 100
model = Chain(Dense(2,1))
mod = FluxModel(model)

generator = GenericGenerator()
```

``` julia
data_train, data_test = Data.train_test_split(counterfactual_data)
Models.train(mod, data_train; n_epochs=n_epochs)
plt_original = plot(mod, counterfactual_data; zoom=0, colorbar=false)
display(plt_original)
```

![](README_files/figure-commonmark/cell-5-output-1.svg)

### Simulation

``` julia
models = Dict(:mymodel => mod)
generators = Dict(:wachter => generator)
experiment = set_up_experiment(data_train, data_test, models, generators)
```

``` julia
run!(experiment)
```

``` julia
new_data = experiment.recourse_systems[1][1].data
new_model = experiment.recourse_systems[1][1].model
plt_original = plot(new_model, new_data; zoom=0, colorbar=false)
```

![](README_files/figure-commonmark/cell-8-output-1.svg)

## Related Research Paper 📝

The package was developed for a research project that investigates the dynamics of various counterfactual generators.

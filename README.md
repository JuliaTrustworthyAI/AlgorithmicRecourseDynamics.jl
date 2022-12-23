
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/dev) [![Build Status](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac) [![Twitter Badge](https://img.shields.io/twitter/url/https/twitter.com/paltmey.svg?style=social&label=Follow%20%40paltmey)](https://twitter.com/paltmey)

# AlgorithmicRecourseDynamics

`AlgorithmicRecourseDynamics.jl` is a Julia package for modeling Algorithmic Recourse Dynamics.

## Related Research Paper 📝

In this work we investigate what happens if Algorithmic Recourse is actually implemented by a large number of individuals. The chart below illustrates what we mean by Endogenous Macrodynamics in Algorithmic Recourse: (a) we have a simple linear classifier trained for binary classification where samples from the negative class (y=0) are marked in blue and samples of the positive class (y=1) are marked in orange; (b) the implementation of AR for a random subset of individuals leads to a noticable domain shift; (c) as the classifier is retrained we observe a corresponding model shift; (d) as this process is repeated, the decision boundary moves away from the target class.

![](paper/www/poc.png)

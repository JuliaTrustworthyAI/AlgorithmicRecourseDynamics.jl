module Models

using AlgorithmicRecourseDynamics
using CounterfactualExplanations

include("utils.jl")

# Models:
include("mlp.jl") # including logistic regression
include("deep_ensemble.jl")

abstract type AbstractTrainableModel end

end
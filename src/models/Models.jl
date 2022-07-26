module Models

include("utils.jl")

# Models:
include("mlp.jl") # including logistic regression
include("laplace.jl")
include("deep_ensemble.jl")

abstract type AbstractTrainableModel end

end
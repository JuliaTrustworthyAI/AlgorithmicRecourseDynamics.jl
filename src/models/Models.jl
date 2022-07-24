module Models

# Models:
include("mlp.jl")
include("laplace.jl")

include("utils.jl")

abstract type AbstractTrainableModel end

end
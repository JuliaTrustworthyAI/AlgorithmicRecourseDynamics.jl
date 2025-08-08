module Models

using AlgorithmicRecourseDynamics
using CounterfactualExplanations


function perturbation(
    model::CounterfactualExplanations.Models.Model, 
    new_model::CounterfactualExplanations.Models.Model
) 
    perturbation(model, new_model, model.type)
end

# Models:
include("mlp.jl") # including logistic regression
include("deep_ensemble.jl")

end

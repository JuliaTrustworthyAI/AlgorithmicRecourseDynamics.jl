using CounterfactualExplanations
using Flux
using LinearAlgebra
using Parameters
using Statistics

function perturbations(
    model::CounterfactualExplanations.Models.Model, 
    new_model::CounterfactualExplanations.Models.Model, 
    type::CounterfactualExplanations.Models.DeepEnsemble
)
    ensemble = model.model
    new_ensemble = new_model.model
    Δ = mean(map(x -> norm(x)/length(x),Flux.params(new_ensemble).-Flux.params(ensemble)))
    return Δ
end

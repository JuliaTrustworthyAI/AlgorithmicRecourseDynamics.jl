using CounterfactualExplanations
using Flux
using LinearAlgebra
using Parameters
using Statistics

function perturbation(
    model::CounterfactualExplanations.Models.FluxEnsemble,
    new_model::CounterfactualExplanations.Models.FluxEnsemble,
)
    ensemble = model.model
    new_ensemble = new_model.model
    Δ = mean(
        map(x -> norm(x) / length(x), Flux.params(new_ensemble) .- Flux.params(ensemble))
    )
    return Δ
end

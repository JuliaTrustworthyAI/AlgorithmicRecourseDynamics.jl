using CounterfactualExplanations
using Flux
using LinearAlgebra
using Parameters
using Statistics

function perturbation(
    model::CounterfactualExplanations.Models.Model, 
    new_model::CounterfactualExplanations.Models.Model, 
    type::CounterfactualExplanations.Models.MLP
)
    mlp = model.model
    new_mlp = new_model.model
    Δ = mean(map(x -> norm(x)/length(x),Flux.params(new_mlp).-Flux.params(mlp)))
    return Δ
end




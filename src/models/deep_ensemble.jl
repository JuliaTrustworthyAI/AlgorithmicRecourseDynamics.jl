using Parameters
@with_kw struct FluxEnsembleParams 
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 10
    data_loader::Function = data_loader
end


import CounterfactualExplanations.Models: FluxEnsemble
function FluxEnsemble(data::CounterfactualData, K::Int=5;kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    input_dim = size(X,1)
    output_dim = length(unique(y))
    output_dim = output_dim==2 ? output_dim=1 : output_dim # adjust in case binary
    ensemble = build_ensemble(K;input_dim=input_dim, output_dim=output_dim,kwargs...)

    if output_dim==1
        M = FluxEnsemble(ensemble; likelihood=:classification_binary)
    else
        M = FluxEnsemble(ensemble; likelihood=:classification_multi)
    end

    return M
end

using Flux
"""
    build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))

Helper function that builds an ensemble of `K` models.
"""
function build_ensemble(K::Int;kwargs...)
    ensemble = [build_mlp(;kwargs...) for i in 1:K]
    return ensemble
end

using Flux
using Flux.Optimise: update!
using CounterfactualExplanations
using CounterfactualExplanations.Models: FluxEnsemble

"""
    train(M::FluxEnsemble, data::CounterfactualData; kwargs...)

Wrapper function to retrain.
"""
function train(M::FluxEnsemble, data::CounterfactualData; kwargs...)

    args = FluxEnsembleParams(; kwargs...)

    # Prepare data:
    data = args.data_loader(data)
    
    # Training:
    ensemble = M.model

    for model in ensemble
        forward!(
            model, data; 
            loss = args.loss,
            opt = args.opt,
            n_epochs = args.n_epochs
        )
    end

    return M
    
end

using LinearAlgebra, Flux, Statistics

function perturbation(model::FluxEnsemble, new_model::FluxEnsemble)
    ensemble = model.model
    new_ensemble = new_model.model
    Δ = mean(map(x -> norm(x)/length(x),Flux.params(new_ensemble).-Flux.params(ensemble)))
    return Δ
end
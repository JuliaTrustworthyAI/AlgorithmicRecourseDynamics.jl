using LaplaceRedux
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

using Parameters
@with_kw struct LaplaceModelParams 
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 10
    data_loader::Function = data_loader
    λ::Real = 1
    H₀::Union{Nothing, AbstractMatrix} = nothing
end

using Flux
using Flux.Optimise: update!
using CounterfactualExplanations
"""
    train(M::LaplaceModel, data::CounterfactualData; kwargs...)

Wrapper function to retrain `LaplaceReduxModel`.
"""
function train(M::LaplaceReduxModel, data::CounterfactualData; kwargs...)

    args = LaplaceModelParams(; kwargs...)

    # Prepare data:
    data = args.data_loader(data)
    
    # Training:
    model = M.model.model
    forward!(
        model, data; 
        loss = args.loss,
        opt = args.opt,
        n_epochs = args.n_epochs
    )

    # Fit Laplace:
    la = Laplace(model, λ=args.λ, H₀=args.H₀)
    LaplaceRedux.fit!(la, data)
    M = LaplaceReduxModel(la)

    return M
    
end

using CounterfactualExplanations.Models: LaplaceReduxModel
function LaplaceReduxModel(data::CounterfactualData;λ=0.1,kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    input_dim = size(X,1)
    output_dim = length(unique(y))
    output_dim = output_dim==2 ? output_dim=1 : output_dim # adjust in case binary
    model = build_mlp(;input_dim=input_dim, output_dim=output_dim,kwargs...)
    model = Laplace(model, λ=λ)

    if output_dim==1
        M = LaplaceReduxModel(model; likelihood=:classification_binary)
    else
        M = LaplaceReduxModel(model; likelihood=:classification_multi)
    end

    return M
end


using LinearAlgebra, Flux, Statistics
function perturbation(model::LaplaceReduxModel, new_model::LaplaceReduxModel; agg=mean)
    mlp = model.model
    new_mlp = new_model.model
    Δ = agg(norm.(collect(Flux.params(new_mlp)) .- collect(Flux.params(mlp))))
    return Δ
end
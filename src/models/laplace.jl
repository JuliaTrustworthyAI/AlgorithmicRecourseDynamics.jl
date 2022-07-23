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
function train(M::LaplaceReduxModel, data::CounterfactualData; τ=nothing, kwargs...)

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
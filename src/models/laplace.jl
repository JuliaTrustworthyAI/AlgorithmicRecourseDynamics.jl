using LaplaceRedux
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

# Step 1)
struct LaplaceModel <: CounterfactualExplanations.Models.AbstractDifferentiableJuliaModel
    model::Laplace
end

# Step 2)
logits(M::LaplaceModel, X::AbstractArray) = M.model.model(X)
probs(M::LaplaceModel, X::AbstractArray)= LaplaceRedux.predict(M.model, X)

using Parameters
@with_kw struct LaplaceModelParams 
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 200
    λ::Real = 1
    H₀::Union{Nothing, AbstractMatrix} = nothing
end

using Flux
using Flux.Optimise: update!
using CounterfactualExplanations
"""
    train(M::LaplaceModel, data::CounterfactualData; kwargs...)

Wrapper function to retrain `LaplaceModel`.
"""
function train(M::LaplaceModel, data::CounterfactualData; kwargs...)

    args = LaplaceModelParams(; kwargs...)

    # Prepare data:
    data = data_loader(data)
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

    # Declare type:
    M = LaplaceModel(la)

    return M
    
end
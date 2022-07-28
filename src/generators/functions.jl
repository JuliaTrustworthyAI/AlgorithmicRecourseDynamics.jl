using LinearAlgebra, CounterfactualExplanations

struct GravitationalGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::Union{AbstractFloat,AbstractVector} # strength of penalty
    ϵ::AbstractFloat # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
using Parameters
@with_kw struct GravitationalGeneratorParams
    ϵ::AbstractFloat=0.1
    τ::AbstractFloat=1e-5
end

"""
    GravitationalGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = GravitationalGenerator()
```
"""
function GravitationalGenerator(;loss::Union{Nothing,Symbol}=nothing,complexity::Function=norm,λ::Union{AbstractFloat,AbstractVector}=[0.1, 1.0],kwargs...)
    params = GravitationalGeneratorParams(;kwargs...)
    GravitationalGenerator(loss, complexity, λ, params.ϵ, params.τ)
end

# Complexity:
using Statistics, LinearAlgebra
function gravity(counterfactual_state::State; K=5)
    ids = rand(1:size(counterfactual_state.params[:potential_neighbours],2),K)
    neighbours = counterfactual_state.params[:potential_neighbours][:,ids]
    centroid = mean(neighbours, dims=2)
    gravity_ = norm(centroid .- counterfactual_state.f(counterfactual_state.s′))
    return gravity_
end

using CounterfactualExplanations.CounterfactualState
import CounterfactualExplanations.Generators: h
"""
    h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(generator::GravitationalGenerator, counterfactual_state::CounterfactualState.State)
    dist_ = generator.complexity(counterfactual_state.x .- counterfactual_state.f(counterfactual_state.s′))
    gravity_ = gravity(counterfactual_state)
    if length(generator.λ)==1
        penalty = generator.λ * (dist_ .+ gravity_)
    else
        penalty = generator.λ[1] * dist_ .+ generator.λ[2] * gravity_
    end
    return penalty
end



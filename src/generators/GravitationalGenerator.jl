using LinearAlgebra, CounterfactualExplanations

mutable struct GravitationalGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::Union{AbstractFloat,AbstractVector} # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat}
    opt::Any # optimizer
    τ::AbstractFloat # tolerance for convergence
    K::Int # number of K randomly chosen neighbours
    centroid::Union{Nothing,AbstractArray}
end

# API streamlining:
using Parameters, Flux
@with_kw struct GravitationalGeneratorParams
    opt::Any=Flux.Optimise.Descent()
    τ::AbstractFloat=1e-5
    K::Int=50
    centroid::Union{Nothing,AbstractArray}=nothing
end

"""
    GravitationalGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        opt::Any=Flux.Optimise.Descent(),
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = GravitationalGenerator()
```
"""
function GravitationalGenerator(
    ;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Function=norm,
    λ::Union{AbstractFloat,AbstractVector}=[0.1,5.0],
    decision_threshold=nothing,
    kwargs...
)
    params = GravitationalGeneratorParams(;kwargs...)
    GravitationalGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ, params.K, params.centroid)
end

# Complexity:
using Statistics, LinearAlgebra
using CounterfactualExplanations.CounterfactualState
import CounterfactualExplanations.Generators: h
"""
    h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(generator::GravitationalGenerator, counterfactual_state::CounterfactualState.State)
    
    # Distance from factual:
    dist_ = generator.complexity(counterfactual_state.x .- counterfactual_state.f(counterfactual_state.s′))

    # Gravitational center:
    if isnothing(generator.centroid)
        ids = rand(1:size(counterfactual_state.params[:potential_neighbours],2),generator.K)
        neighbours = counterfactual_state.params[:potential_neighbours][:,ids]
        generator.centroid = mean(neighbours, dims=2)
    end

    # Distance from gravitational center:
    gravity_ = generator.complexity(generator.centroid .- counterfactual_state.f(counterfactual_state.s′))
    
    if length(generator.λ)==1
        penalty = generator.λ * (dist_ .+ gravity_)
    else
        penalty = generator.λ[1] * dist_ .+ generator.λ[2] * gravity_
    end
    return penalty
end



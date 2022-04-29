"""
    Grid(μ::AbstractArray, γ::AbstractArray, ℓ::AbstractArray)

Sets up the the grid of variables. `μ` refers to the proportion of individuals that shall
recieve recourse and `γ` refers to the desired threshold probability for recourse.
"""
struct GridVariables
    μ::AbstractArray
    γ::AbstractArray
end

"""
    Grid(grid::Base.Iterators.ProductIterator)

The unfolded grid containing all combinations of all variables.
"""
struct Grid
    grid::Base.Iterators.ProductIterator
end

"""
    build_grid(grid::GridVariables) = Base.Iterators.product(grid.μ, grid.γ)    

Builds the grid based on provided variables.
"""
build_grid(grid::GridVariables) = Grid(Base.Iterators.product(grid.μ, grid.γ))


using CounterfactualExplanations

"""
    Experiment(X::AbstractArray,y::AbstractArray,𝑴::CounterfactualExplanations.AbstractFittedModel,target::AbstractFloat,grid::Base.Iterators.ProductIterator,n_rounds::Int)

Sets up the experiment to be run.
"""
struct Experiment
    X::AbstractArray
    y::AbstractArray
    𝑴::CounterfactualExplanations.AbstractFittedModel
    target::AbstractFloat
    grid::GridVariables
    n_rounds::Int
end

using Random, StatsBase, LinearAlgebra, Flux
using ..Models
using CounterfactualExplanations
using CounterfactualExplanations.Counterfactuals: counterfactual, counterfactual_label
"""
    run_experiment(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run_experiment(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000, τ=1.0, store_path=false)

    # Setup:
    if !isnothing(seed)
        Random.seed!(seed)
    end
    output = []
    path = []
    grid = build_grid(experiment.grid)

    for k in 1:n_folds
        for (μ,γ) in grid.grid

            X = copy(experiment.X)
            y = copy(experiment.y)
            chosen_individuals = []
            
            for t in 1:experiment.n_rounds

                # Data: 
                data = Models.prepare_data(X,y)
                counterfactual_data = CounterfactualData(X,y')
                
                # Classifier:
                if t > 1
                    𝑴 = Models.retrain(experiment.𝑴, data, τ=τ)
                else
                    𝑴 = experiment.𝑴
                end
    
                # Choose individuals:
                adverse_outcome = findall(vec(experiment.y .!=  experiment.target))
                n_individuals = Int(round(μ * length(adverse_outcome)))
                chosen_individualsₜ = StatsBase.sample(adverse_outcome,n_individuals,replace=false)

                # Generate recourse:
                for i in chosen_individualsₜ
                    x = X[:,i]
                    outcome = generate_counterfactual(x, experiment.target, counterfactual_data, 𝑴, generator; T=T)
                    X[:,i] = counterfactual(outcome) # update individuals features
                    y[i] = counterfactual_label(outcome)
                end

                # Evaluate recourse:
                chosen_individuals = union(chosen_individuals, chosen_individualsₜ)
                pct_validₜ = sum(y[chosen_individuals] .== experiment.target)/length(chosen_individuals)
                ΔX = X[:,chosen_individuals] .- experiment.X[:,chosen_individuals]
                avg_costₜ = mean(norm.(ΔX, 2))

                # Collect and store output:
                outputₜ = (
                    pct_valid=pct_validₜ, 
                    avg_cost=avg_costₜ,
                    t = t,
                    μ = μ,
                    γ = γ,
                    k = k
                )
                output = vcat(output, outputₜ)

                if store_path
                    pathₜ = (
                        X̲ = copy(X),
                        y̲ = copy(y),
                        𝑴 = 𝑴,
                        t = t,
                        μ = μ,
                        γ = γ,
                        k = k
                    )
                    path = vcat(path, pathₜ)
                end

            end
        end
    end

    return output, path

end


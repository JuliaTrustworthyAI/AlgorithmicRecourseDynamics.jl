using CounterfactualExplanations

using Parameters
@with_kw mutable struct FixedParamters
    n_rounds::Int = 5
    n_folds::Int = 5
    seed::Union{Nothing, Int} = nothing
    T::Int = 1000
    μ::AbstractFloat = 0.05
    γ::AbstractFloat = 0.75
    intersect_::Bool = true
end

mutable struct Experiment
    data::CounterfactualExplanations.CounterfactualData
    target::Number
    grid::AbstractArray
    fixed_parameters::Union{Nothing,FixedParamters}
end

function Experiment(data::CounterfactualExplanations.CounterfactualData, target::Number, models::NamedTuple, generators::NamedTuple)
    
    # Set up grid
    grid = Base.Iterators.product(models, generators)
    grid = map(grid) do vars
        model = vars[1]
        generator = vars[2]
        recourse_system = RecourseSystem(data, model, generator)
        grid_element = Dict(
            :model => model,
            :recourse_system => recourse_system
        )
        return grid_element
    end

    experiment = Experiment(
        data,
        target,
        grid,
        nothing
    )

    return experiment
end


using CounterfactualExplanations

"""
    Experiment(X::AbstractArray,y::AbstractArray,M::CounterfactualExplanations.AbstractFittedModel,target::AbstractFloat,grid::Base.Iterators.ProductIterator,n_rounds::Int)

Sets up the experiment to be run.
"""
mutable struct RecourseSystem
    data::CounterfactualExplanations.CounterfactualData
    model::CounterfactualExplanations.AbstractFittedModel
    generator::CounterfactualExplanations.Generators.AbstractGenerator
end

using StatsBase
"""
    choose_individuals(system::RecourseSystem, target::Number)
    
"""
function choose_individuals(experiment::Experiment; intersect_::Bool=true)
    args = experiment.fixed_parameters
    target, μ = experiment.target, args.μ

    candidates = map(experiment.grid) do x
        recourse_system = x[:recourse_system]
        findall(vec(recourse_system.data.y .!= target))
    end

    if intersect_
        candidates_intersect = intersect(candidates...)
        n_individuals = Int(round(μ * length(candidates_intersect)))
        chosen_individuals = StatsBase.sample(candidates_intersect,n_individuals,replace=false)
        chosen_individuals = map(candidates) do x
            sort(chosen_individuals)
        end
    else
        chosen_individuals = map(candidates) do x
            n_individuals = Int(round(μ * length(x)))
            sort(StatsBase.sample(x,n_individuals,replace=false))
        end
    end

    return chosen_individuals
end

using CounterfactualExplanations.DataPreprocessing: unpack
using CounterfactualExplanations
using CounterfactualExplanations.Counterfactuals: counterfactual, counterfactual_label
"""

"""
function update!(experiment::Experiment, system::RecourseSystem, chosen_individuals::AbstractVector)
    
    # Recourse System:
    counterfactual_data = system.data
    X, y = unpack(counterfactual_data)
    M = system.model
    generator = system.generator

    # Generate recourse:
    for i in chosen_individuals
        x = X[:,i]
        outcome = generate_counterfactual(x, experiment.target, counterfactual_data, M, generator; T=experiment.T, γ=experiment.γ)
        X[:,i] = counterfactual(outcome) # update individuals' features
        y[:,i] = first(counterfactual_label(outcome)) # update individuals' predicted label
    end

    # Update data and classifier:
    system.newdata = CounterfactualData(X,y)
    system.newmodel = AlgorithmicRecourseDynamics.Models.train(system.newmodel, counterfactual_data; τ=experiment.τ)
end



using Random, StatsBase, LinearAlgebra, Flux
"""
    run_experiment(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run_experiment(experiment::Experiment; store_path=true, fixed_parameters...)

    # Load fixed hyperparameters:
    args = FixedParamters(fixed_parameters...)
    experiment.fixed_parameters = args
    K, N, γ, μ, intersect_ = args.n_folds, args.n_rounds, args.γ, args.μ, args.intersect_
    M = length(experiment.grid)

    # Setup:
    if !isnothing(experiment.seed)
        Random.seed!(experiment.seed)
    end

    # Pre-allocate memory:
    output = []

    for k in 1:K
        for n in 1:N

            # Choose individuals that shall receive recourse:
            chosen_individuals = choose_individuals(experiment; intersect_=intersect_)

            for m in 1:M
                
                element = experiment.grid[m]
                chosen_individuals_m = chosen_individuals[m]
                recourse_system = element[:recourse_system]

                # Update experiment
                update!(recourse_system, experiment, chosen_individuals_m)

                # Evaluate:
                eval_ = evaluate_system(recourse_system, experiment)

                # Store:
                if store_path
                    output_ = (
                        eval_ = eval_,
                        m = m,
                        k = k,
                        n = n,
                        recourse_system = recourse_system
                    )
                else
                    output_ = (
                        eval_ = eval_,
                        m = m,
                        k = k,
                        n = n
                    )
                end

                output = vcat(output, output_)
               
            end
        end
    end

    return output

end

using BSON
"""
    save_path(root,path)

Helper function to save `path` output from `run_experiment` to BSON.
"""
function save_path(root,path)
    bson(root * "_path.bson",Dict(i => path[i] for i ∈ 1:length(path)))
end

using BSON
"""
    load_path(root,path)

Helper function to load `path` output.
"""
function load_path(root)
    dict = BSON.load(root * "_path.bson")
    path = [dict[i] for i ∈ 1:length(dict)]
    return path
end


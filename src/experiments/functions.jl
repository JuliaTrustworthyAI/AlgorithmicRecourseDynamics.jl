using CounterfactualExplanations

using Parameters
@with_kw mutable struct FixedParameters
    n_rounds::Int = 5
    n_folds::Int = 5
    seed::Union{Nothing, Int} = nothing
    T::Int = 1000
    μ::AbstractFloat = 0.05
    γ::AbstractFloat = 0.75
    intersect_::Bool = true
    τ::AbstractFloat = 1.0
end

mutable struct Experiment
    data::CounterfactualExplanations.CounterfactualData
    target::Number
    recourse_systems::AbstractArray
    system_identifiers::Base.Iterators.ProductIterator
    fixed_parameters::Union{Nothing,FixedParameters}
end

"""
    Experiment(data::CounterfactualExplanations.CounterfactualData, target::Number, models::NamedTuple, generators::NamedTuple)


"""
function Experiment(data::CounterfactualExplanations.CounterfactualData, target::Number, models::NamedTuple, generators::NamedTuple)
    
    # Set up grid
    grid = Base.Iterators.product(models, generators)
    recourse_systems = map(grid) do vars
        newdata = deepcopy(data)
        model = vars[1] # initial model is owned by the recourse systems
        newmodel = deepcopy(model)
        generator = vars[2]
        recourse_system = RecourseSystem(newdata, newmodel, generator, model)
        return recourse_system
    end

    # Add system identifiers:
    system_identifiers = Base.Iterators.product(keys(models), keys(generators))

    experiment = Experiment(
        data, # initial data is owned by the experiment, shared across recourse systems
        target,
        recourse_systems,
        system_identifiers,
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
    initial_model::CounterfactualExplanations.AbstractFittedModel
end

using StatsBase
"""
    choose_individuals(system::RecourseSystem, target::Number)
    
"""
function choose_individuals(experiment::Experiment; intersect_::Bool=true)
    args = experiment.fixed_parameters
    target, μ = experiment.target, args.μ

    candidates = map(experiment.recourse_systems) do x
        findall(vec(x.data.y .!= target))
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
using ..Models
"""

"""
function update!(experiment::Experiment, recourse_system::RecourseSystem, chosen_individuals::AbstractVector)
    
    # Recourse System:
    counterfactual_data = recourse_system.data
    X, y = unpack(counterfactual_data)
    M = recourse_system.model
    generator = recourse_system.generator

    # Experiment:
    args = experiment.fixed_parameters
    T, γ, τ = args.T, args.γ, args.τ
    target = experiment.target

    # Generate recourse:
    for i in chosen_individuals
        x = X[:,i]
        outcome = generate_counterfactual(x, target, counterfactual_data, M, generator; T=T, γ=γ)
        X[:,i] = counterfactual(outcome) # update individuals' features
        y[:,i] .= first(counterfactual_label(outcome)) # update individuals' predicted label
    end

    # Update data and classifier:
    recourse_system.data = CounterfactualData(X,y)
    recourse_system.model = Models.train(M, counterfactual_data; τ=τ)
end




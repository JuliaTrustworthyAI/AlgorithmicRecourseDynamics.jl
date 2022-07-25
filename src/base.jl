using .Experiments: Experiment, FixedParameters, set_up_system_grid!, update!, choose_individuals
using .Evaluation: evaluate_system
using Random, StatsBase, LinearAlgebra, Flux
using DataFrames
using ProgressMeter
using Logging
using Statistics
using CounterfactualExplanations

is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

"""
    run!(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run!(
    experiment::Experiment; evaluate_every=10, n_boostrap=1000, forward=false, show_progress=!is_logging(stderr), fixed_parameters...
)

    # Load fixed hyperparameters:
    args = FixedParameters(;fixed_parameters...)
    experiment.fixed_parameters = args
    K, N, intersect_ = args.n_folds, args.n_rounds, args.intersect_
    M = length(experiment.system_identifiers)

    # Setup:
    if !isnothing(args.seed)
        Random.seed!(args.seed)
    end
    if !forward
        set_up_system_grid!(experiment, K)
    else
        @assert !isnothing(experiment.recourse_systems) "Cannot forward an experiment that has never been run."
    end

    # Pre-allocate memory:
    output = [DataFrame() for i in 1:M]

    p_fold = Progress(K; desc="Progress on folds:", showspeed=true, enabled=show_progress, output = stderr)
    @info "Running experiment ..."
    for k in 1:K
        recourse_systems = experiment.recourse_systems[k]
        chosen_individuals = zeros(size(recourse_systems))
        p_round = Progress(N; desc="Progress on rounds:", showspeed=true, enabled=show_progress, output = stderr)
        for n in 1:N
            # Choose individuals that shall receive recourse:
            chosen_individuals_n = choose_individuals(experiment, recourse_systems; intersect_=intersect_)
            chosen_individuals = map((x,y) -> union(x,y),chosen_individuals,chosen_individuals_n)
            Threads.@threads for m in 1:M
                recourse_system = recourse_systems[m]
                chosen_individuals_m = chosen_individuals_n[m]
                recourse_systems[m].chosen_individuals = chosen_individuals[m]
                with_logger(NullLogger()) do
                    # Update experiment
                    update!(experiment, recourse_system, chosen_individuals_m)
                end
                # Evaluate:
                if n % evaluate_every == 0 
                    evaluation = evaluate_system(recourse_system, experiment, n=n_boostrap)
                    # Store results:
                    evaluation.k .= k
                    evaluation.n .= n
                    evaluation.model .= collect(experiment.system_identifiers)[m][1]
                    evaluation.generator .= collect(experiment.system_identifiers)[m][2]
                    evaluation.n_individuals .= length(chosen_individuals[m])
                    evaluation.pct_total .= length(chosen_individuals[m])/size(experiment.train_data.y,2)
                    # Add recourse measures:
                    bmk = mapcols(mean, recourse_system.benchmark)
                    evaluation.success_rate .= bmk.success_rate
                    evaluation.distance .= bmk.distance
                    evaluation.redundancy .= bmk.redundancy

                    output[m] = vcat(output[m], evaluation)
                end
            end
            next!(p_round, showvalues = [(:Fold, k), (:Round, n)])
        end
        next!(p_fold)
    end

    # Collect output:
    output = reduce(vcat, output)

    return output

end

struct ExperimentResults 
    output::DataFrame
    experiment::Experiment
end

using DataFrames, CSV, BSON
function run_experiment(
    data::CounterfactualData,
    models::Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    evaluate_every::Int=2,
    pre_train_models::Union{Nothing,Int}=100,
    save_path::Union{Nothing,String}=nothing,
    save_name::Union{Nothing,String}=nothing,
    kwargs...
)

    @info "Starting experiment"

    # Data:
    data_train, data_test = Models.train_test_split(data)

    # Pretrain:
    if !isnothing(pre_train_models)
        map!(model -> Models.train(model, data_train; n_epochs=pre_train_models), values(models))
    end

    # Run:
    experiment = Experiment(data_train, data_test, target, models, deepcopy(generators), num_counterfactuals)
    output = run!(experiment; evaluate_every=evaluate_every, kwargs...)

    @info "Completed experiment."

    results = Experiment(output,experiment)

    # Save to disk:
    if !isnothing(save_path)
        save_name = isnothing(save_name) ? "experiment" : "experiment_$(save_name)"
        save_path = joinpath(save_path,save_name)
        mkpath(save_path)
        CSV.write(joinpath(save_path,"output.csv"), output)
        BSON.@save joinpath(save_path,"output.bson") output
        BSON.@save joinpath(save_path,"experiment.bson") experiment
        BSON.@save joinpath(save_path,"results.bson") results
    end

    return results
    
end

function run_experiment(
    data::CounterfactualData,
    models::Vector{Symbol},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    evaluate_every::Int=2,
    pre_train_models::Union{Nothing,Int}=100,
    save_path::Union{Nothing,String}=nothing,
    save_name::Union{Nothing,String}=nothing,
    kwargs...
)   

    available_models = [:FluxModel, :LaplaceReduxModel]
    @assert all(map(model -> model in available_models, models)) "`models` can only be $(available_models)"

    models = Dict([(model,getfield(AlgorithmicRecourseDynamics.Models, model)(data)) for model in models])

    output = run_experiment(
        data, models, generators;
        target=target,num_counterfactuals=num_counterfactuals,pre_train_models=100,
        save_path=save_path,save_name=save_name,evaluate_every=evaluate_every,
        kwargs...
    )

    return output

end


function run_experiments(
    catalogue::Dict{Symbol, CounterfactualData},
    models::Union{Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},Vector{Symbol}},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    evaluate_every::Int=2,
    pre_train_models::Union{Nothing, Int}=100,
    save_path::Union{Nothing,String}=nothing,
    kwargs...
)

    run_single(data, save_name) = run_experiment(
        data, models, generators;
        target=target, num_counterfactuals=num_counterfactuals,
        evaluate_every=evaluate_every,
        pre_train_models=pre_train_models,
        save_path=save_path,
        save_name=save_name,
        kwargs...
    )

    output = [run_single(data,string(name)) for (name,data) in catalogue]
    
    return output
end


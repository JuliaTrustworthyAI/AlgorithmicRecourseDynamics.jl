using .Experiments: Experiment, FixedParameters, RecourseSystem, set_up_system_grid!, update!, choose_individuals
using .Evaluation: evaluate_system
using Random, StatsBase, LinearAlgebra, Flux
using DataFrames
using ProgressMeter
using Logging
using Statistics
using CounterfactualExplanations
using Serialization

is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

function collect_output(
    experiment::Experiment, recourse_system::RecourseSystem, chosen_individuals::Union{Nothing,AbstractArray}, k::Int, n::Int, m::Int;
    n_bootstrap=1000
)

    # Evaluate:
    output = evaluate_system(recourse_system, experiment, n=n_bootstrap)
    
    # Add additional information:
    output.k .= k
    output.n .= n
    output.model .= collect(experiment.system_identifiers)[m][1]
    output.generator .= collect(experiment.system_identifiers)[m][2]
    output.n_individuals .= isnothing(chosen_individuals) ? 0 : length(chosen_individuals)
    output.pct_total .= isnothing(chosen_individuals) ? 0 : length(chosen_individuals)/size(experiment.train_data.y,2)

    # Add recourse measures:
    if n > 0 
        bmk = mapcols(mean, recourse_system.benchmark)
        output.success_rate .=  bmk.success_rate
        output.distance .= bmk.distance
        output.redundancy .= bmk.redundancy
    end

    return output
end

"""
    run!(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run!(
    experiment::Experiment; evaluate_every=10, n_bootstrap=1000, forward=false, show_progress=!is_logging(stderr), fixed_parameters...
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
        # Initial evaluation:
        for m in 1:M
            output_initial = collect_output(experiment, recourse_systems[m], nothing, k, 0, m, n_bootstrap=nothing)
            output[m] = vcat(output[m], output_initial, cols=:union)
        end
        # Recursion over N rounds:
        chosen_individuals = zeros(size(recourse_systems))
        p_round = Progress(N; desc="Progress on rounds:", showspeed=true, enabled=show_progress, output = stderr)
        for n in 1:N
            # Choose individuals that shall receive recourse:
            chosen_individuals_n = choose_individuals(experiment, recourse_systems; intersect_=intersect_)
            chosen_individuals = map((x,y) -> union(x,y),chosen_individuals,chosen_individuals_n)
            Threads.@threads for m in 1:M
                recourse_system = recourse_systems[m]
                chosen_individuals_m = chosen_individuals_n[m]
                recourse_system.chosen_individuals = chosen_individuals[m]
                # Update experiment
                with_logger(NullLogger()) do
                    update!(experiment, recourse_system, chosen_individuals_m)
                end
                # Evaluate:
                if n % evaluate_every == 0 
                    output_checkpoint = collect_output(experiment, recourse_system, chosen_individuals[m], k, n, m, n_bootstrap=n_bootstrap)
                    output[m] = vcat(output[m], output_checkpoint, cols=:union)
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

"""
    set_up_experiment(
        data::CounterfactualData,
        models::Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},
        generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
        target::Int=1,
        num_counterfactuals::Int=5,
        pre_train_models::Union{Nothing,Int}=100,
        kwargs...
    )
    
Sets up one experiment for the provided data, models and generators.
"""
function set_up_experiment(
    data_train::CounterfactualData,
    data_test::CounterfactualData,
    models::Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    kwargs...
)

    experiment = Experiment(data_train, data_test, target, models, deepcopy(generators), num_counterfactuals)

    # Sanity check:
    @info "Initial model scores:"
    println(experiment.initial_model_scores)

    return experiment
    
end

"""
    set_up_experiment(
        data::CounterfactualData,
        models::Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},
        generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
        target::Int=1,
        num_counterfactuals::Int=5,
        pre_train_models::Union{Nothing,Int}=100,
        kwargs...
    )
    
Sets up one experiment for the provided data, models and generators.
"""
function set_up_experiment(
    data::CounterfactualData,
    models::Vector{Symbol},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    model_params::NamedTuple=(batch_norm=false,dropout=false,activation=Flux.relu),
    target::Int=1,
    num_counterfactuals::Int=5,
    pre_train_models::Union{Nothing,Int}=100,
    kwargs...
)

    available_models = [:LogisticRegression, :FluxModel, :FluxEnsemble, :LaplaceReduxModel]
    @assert all(map(model -> model in available_models, models)) "`models` can only be $(available_models)"

    models = Dict([(model,getfield(AlgorithmicRecourseDynamics.Models, model)(data; model_params...)) for model in models])

    # Data:
    data_train, data_test = Models.train_test_split(data)

    # Pretrain:
    if !isnothing(pre_train_models)
        map!(model -> Models.train(model, data_train; n_epochs=pre_train_models, kwargs...), values(models))
    end

    experiment = Experiment(data_train, data_test, target, models, deepcopy(generators), num_counterfactuals)

    # Sanity check:
    @info "Initial model scores:"
    println(experiment.initial_model_scores)
    
    return experiment

end


"""
    function set_up_experiments(
        catalogue::Dict{Symbol, CounterfactualData},
        models::Union{Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},Vector{Symbol}},
        generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
        target::Int=1,
        num_counterfactuals::Int=5,
        pre_train_models::Union{Nothing, Int}=100,
        kwargs...
    )

Sets up multiple experiments.
"""
function set_up_experiments(
    catalogue::Dict{Symbol, CounterfactualData},
    models::Union{Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},Vector{Symbol}},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    pre_train_models::Union{Nothing, Int}=100,
    kwargs...
)
    set_up_single(data) = set_up_experiment(
        data, models, generators;
        target=target, num_counterfactuals=num_counterfactuals,
        pre_train_models=pre_train_models,
        kwargs...
    )

    experiments = Dict(key => set_up_single(data) for (key,data) in catalogue)

    return experiments
end

struct ExperimentResults 
    output::DataFrame
    experiment::Experiment
end

using DataFrames, CSV, BSON
"""
    run_experiment(
        experiment::Experiment; evaluate_every::Int=2,
        save_path::Union{Nothing,String}=nothing,
        save_name::Union{Nothing,String}=nothing,
        kwargs...
    )

Runs a given experiment and saves the results if specified.
"""
function run_experiment(
    experiment::Experiment; 
    evaluate_every::Int=2,
    save_path::Union{Nothing,String}=nothing,
    save_name::Union{Nothing,String}=nothing,
    kwargs...
)

    exp_name = isnothing(save_name) ? "unnamed" : save_name

    @info "Starting experiment: $exp_name"

    # Run:
    output = run!(experiment; evaluate_every=evaluate_every, kwargs...)

    @info "Completed experiment: $exp_name"

    results = ExperimentResults(output,experiment)

    # Save to disk:
    if !isnothing(save_path)
        save_name = isnothing(save_name) ? "experiment" : "experiment_$(save_name)"
        save_path = joinpath(save_path,save_name)
        mkpath(save_path)
        CSV.write(joinpath(save_path,"output.csv"), output)
        Serialization.serialize(joinpath(save_path,"output.jls"), output)
        Serialization.serialize(joinpath(save_path,"experiment.jls"), experiment)
        Serialization.serialize(joinpath(save_path,"results.jls"), results)

        @info "Saved experiment: $exp_name"

    end

    return results
end

"""
    run_experiment(
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

Sets up one experiment for the provided data, models and generators and then runs it. Saves results if specified. Models and generators need to be supplied as dictionaries, where values need to be of type `CounterfactualExplanations.Models.AbstractFittedModel` and `CounterfactualExplanations.Generators.AbstractGenerator`, respectively.
"""
function run_experiment(
    data::CounterfactualData,
    models::Union{Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},Vector{Symbol}},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    evaluate_every::Int=2,
    pre_train_models::Union{Nothing,Int}=100,
    save_path::Union{Nothing,String}=nothing,
    save_name::Union{Nothing,String}=nothing,
    kwargs...
)

    experiment = set_up_experiment(
        data,models,generators;
        target=target,num_counterfactuals=num_counterfactuals,pre_train_models=pre_train_models
    )

    exp_name = isnothing(save_name) ? "unnamed" : save_name

    @info "Starting experiment: $exp_name"

    # Run:
    output = run!(experiment; evaluate_every=evaluate_every, kwargs...)

    @info "Completed experiment: $exp_name"

    results = ExperimentResults(output,experiment)

    # Save to disk:
    if !isnothing(save_path)
        save_name = isnothing(save_name) ? "experiment" : "experiment_$(save_name)"
        save_path = joinpath(save_path,save_name)
        mkpath(save_path)
        CSV.write(joinpath(save_path,"output.csv"), output)
        Serialization.serialize(joinpath(save_path,"output.jls"), output)
        Serialization.serialize(joinpath(save_path,"experiment.jls"), experiment)
        Serialization.serialize(joinpath(save_path,"results.jls"), results)

        @info "Saved experiment: $exp_name"

    end

    return results
    
end

"""
    function run_experiments(
        experiments::Dict{Symbol, Experiment};
        evaluate_every::Int=2,
        save_path::Union{Nothing,String}=nothing,
        kwargs...
    )

Runs multiple provided experiments.
"""
function run_experiments(
    experiments::Dict{Symbol, Experiment};
    evaluate_every::Int=2,
    save_path::Union{Nothing,String}=nothing,
    save_name_suffix::String="",
    create_copy::Bool=true,
    kwargs...
)

    if create_copy
        experiments = deepcopy(experiments)
    end

    run_single(experiment, name) = run_experiment(
        experiment;
        evaluate_every=evaluate_every,
        save_path=save_path,
        save_name=name,
        kwargs...
    )

    save_name_suffix = save_name_suffix != "" ? "_$save_name_suffix" : save_name_suffix
    output = Dict(name => run_single(experiment,"$(string(name))$(save_name_suffix)") for (name,experiment) in experiments)

    return output
    
end

"""
    run_experiments(
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

Sets up and runs experiments for multiple data sets.
"""
function run_experiments(
    catalogue::Dict{Symbol, CounterfactualData},
    models::Union{Dict{Symbol, <: CounterfactualExplanations.Models.AbstractFittedModel},Vector{Symbol}},
    generators::Dict{Symbol, <: CounterfactualExplanations.Generators.AbstractGenerator};
    target::Int=1,
    num_counterfactuals::Int=5,
    evaluate_every::Int=2,
    pre_train_models::Union{Nothing, Int}=100,
    save_path::Union{Nothing,String}=nothing,
    save_name_suffix::String="",
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

    save_name_suffix = save_name_suffix != "" ? "_$save_name_suffix" : save_name_suffix
    output = Dict(name => run_single(data,"$(string(name))$(save_name_suffix)") for (name,data) in catalogue)
    
    return output
end


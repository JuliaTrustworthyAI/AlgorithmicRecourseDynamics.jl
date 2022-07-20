using Random, StatsBase, LinearAlgebra, Flux
using .Experiments: Experiment, FixedParameters, choose_individuals, update!, set_up_system_grid!
using .Evaluation: evaluate_system
using DataFrames
using ProgressMeter
using Logging

is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

"""
    run_experiment(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run_experiment(experiment::Experiment; evaluate_every=10, store_path=true, forward=false, show_progress=!is_logging(stderr), fixed_parameters...)

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
    output = DataFrame()

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
                    evaluation = evaluate_system(recourse_system, experiment)
                    # Store results:
                    evaluation.k .= k
                    evaluation.n .= n
                    evaluation.model .= collect(experiment.system_identifiers)[m][1]
                    evaluation.generator .= collect(experiment.system_identifiers)[m][2]
                    evaluation.n_individuals .= length(chosen_individuals[m])
                    evaluation.pct_total .= length(chosen_individuals[m])/size(experiment.data.y,2)
                    output = vcat(output, evaluation)
                end
            end
            if size(output,1) > 0
                next!(p_round, showvalues = [(:Fold, k), (:Round, n), (:output, output[(output.k .== k) .& (output.n .== n),:])])
            else
                next!(p_round, showvalues = [(:Fold, k), (:Round, n)])
            end
        end
        next!(p_fold)
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


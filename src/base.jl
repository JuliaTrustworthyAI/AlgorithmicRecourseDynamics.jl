using Random, StatsBase, LinearAlgebra, Flux
using .Experiments: Experiment, FixedParameters, choose_individuals, update!
using .Evaluation: evaluate_system
using DataFrames
using ProgressBars
"""
    run_experiment(experiment::Experiment, generator::CounterfactualExplanations.AbstractGenerator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run_experiment(experiment::Experiment; store_path=true, fixed_parameters...)

    # Load fixed hyperparameters:
    args = FixedParameters(;fixed_parameters...)
    experiment.fixed_parameters = args
    K, N, intersect_ = args.n_folds, args.n_rounds, args.intersect_
    M = length(experiment.recourse_systems)

    # Setup:
    if !isnothing(args.seed)
        Random.seed!(args.seed)
    end

    # Pre-allocate memory:
    output = DataFrame()
    chosen_individuals = zeros(size(experiment.recourse_systems))

    fold_iter = ProgressBar(1:K)
    round_iter = ProgressBar(1:N)
    system_iter = ProgressBar(1:M)
    for k in fold_iter
        set_description(fold_iter, string("Fold $k out of $K"))
        for n in round_iter
            set_description(round_iter, string("Round $n out of $N"))

            # Choose individuals that shall receive recourse:
            chosen_individuals_n = choose_individuals(experiment; intersect_=intersect_)
            chosen_individuals = map((x,y) -> union(x,y),chosen_individuals,chosen_individuals_n)

            Threads.@threads for m in system_iter
                set_description(system_iter, string("Recourse System $m out of $M (on thread $(Threads.threadid()))"))
                recourse_system = experiment.recourse_systems[m]
                chosen_individuals_m = chosen_individuals_n[m]

                # Update experiment
                update!(experiment, recourse_system, chosen_individuals_m)

                # Evaluate:
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

            println("Output for fold $k, round $n:")
            println(output[(output.k .== k) .& (output.n .== n),:])

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


using CounterfactualExplanations.Models: probs
using DataFrames
using LinearAlgebra
using ..Models: model_evaluation
using StatsBase

struct ModelMetric <: AbstractMetric
    metric::Number
    p_value::Union{Missing,Number}
    name::Symbol
end

"""
    DataFrame(metric::ModelMetric)

Turns an instance of class `ModelMetric` into `DataFrame`.
"""
function DataFrames.DataFrame(metric::ModelMetric)
    vals = (metric.metric, metric.p_value, metric.name, :model)
    df = DataFrame(NamedTuple{(:value, :p_value, :name, :scope)}.([vals]))
    return df
end

function perturbation(experiment::Experiment, recourse_system::RecourseSystem)

    # Initial:
    M = recourse_system.initial_model
    # New:
    new_M = recourse_system.model

    value = perturbation(M, new_M)

    metric = ModelMetric(value, missing, :perturbation)

    return metric
end

"""
    mmd_model(experiment::Experiment, recourse_system::RecourseSystem; n=1000, grid_search=false, kwargs...)

Calculates the MMD on the probabilities of classification assigned by the model to the set of (all) instances. Allows to quantify the model shift.
"""
function mmd_model(
    experiment::Experiment,
    recourse_system::RecourseSystem;
    n=1000,
    grid_search=false,
    n_samples=1000,
    kwargs...,
)
    X, _ = CounterfactualExplanations.DataPreprocessing.unpack_data(experiment.data)

    if grid_search
        X = reduce(
            hcat,
            [
                map(x -> rand(range(x...; length=100)), extrema(X; dims=2)) for
                i in 1:n_samples
            ],
        )
    end

    # Initial:
    M = recourse_system.initial_model
    proba = probs(M, X)

    # New:
    new_M = recourse_system.model
    new_proba = probs(new_M, X)

    value, p_value = mmd(proba, new_proba, n_samples; compute_p=n, kwargs...)
    metric_name = grid_search ? :mmd_grid : :mmd

    metric = ModelMetric(value, p_value, metric_name)

    return metric
end

"""
    decisiveness(experiment::Experiment, recourse_system::RecourseSystem)

Calculates the pseudo-distance of points to the decision boundary measured as the average probability of classification centered around 0.5. High value corresponds to a large margin of classification.
"""
function decisiveness(experiment::Experiment, recourse_system::RecourseSystem)
    X, _ = CounterfactualExplanations.DataPreprocessing.unpack_data(experiment.data)

    # Initial:
    M = recourse_system.initial_model
    proba = probs(M, X)

    # New:
    new_M = recourse_system.model
    new_proba = probs(new_M, X)

    value = abs(norm(proba .- 0.5) - norm(new_proba .- 0.5))

    metric = ModelMetric(value, missing, :decisiveness)

    return metric
end

"""
    disagreement(experiment::Experiment, recourse_system::RecourseSystem)    

Calculates the Disagreement pseudo-distance defined in https://doi.org/10.1145/1273496.1273541 as Pr(h(x) != h'(x)), that is the probability that labels assigned by one classifier do not agree with the labels assigned by another classifier. Simply put, it measures the overlap between models. As this is an empirical measure, we can vary the number of records in `data`.
"""
function disagreement(experiment::Experiment, recourse_system::RecourseSystem)
    X, _ = CounterfactualExplanations.DataPreprocessing.unpack_data(experiment.data)

    # Initial:
    M = recourse_system.initial_model
    proba = reduce(hcat, map(x -> length(x) == 1 ? [x, 1 - x] : x, probs(M, X)))
    # New:
    new_M = recourse_system.model
    new_proba = reduce(hcat, map(x -> length(x) == 1 ? [x, 1 - x] : x, probs(new_M, X)))

    value = sum(argmax(proba; dims=1) .!= argmax(new_proba; dims=1)) / size(X, 2)
    metric = ModelMetric(value, missing, :disagreement)

    return metric
end

function model_performance(experiment::Experiment, recourse_system::RecourseSystem)

    # Initial:
    M = recourse_system.initial_model
    score_ = model_evaluation(M, experiment.test_data)
    @assert score_ == recourse_system.initial_score

    # New:
    new_M = recourse_system.model
    new_score_ = model_evaluation(new_M, experiment.test_data)

    # Difference:
    value = new_score_ - score_
    metric = ModelMetric(value, missing, :model_performance)
    return metric
end

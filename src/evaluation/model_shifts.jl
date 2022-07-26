struct ModelMetric <: AbstractMetric 
    metric::Number
    p_value::Union{Missing,Number}
    name::Symbol
end

using DataFrames
import DataFrames: DataFrame
"""
    DataFrame(metric::ModelMetric)

Turns an instance of class `ModelMetric` into `DataFrame`.
"""
function DataFrame(metric::ModelMetric)
    vals = (metric.metric, metric.p_value, metric.name, :model)
    df = DataFrame(NamedTuple{(:value, :p_value, :name, :scope)}.([vals]))
    return df  
end

using LinearAlgebra
function perturbation(experiment::Experiment, recourse_system::RecourseSystem)

    # Initial:
    M = recourse_system.initial_model
    # New:
    new_M = recourse_system.model

    value = perturbation(M, new_M)

    metric = ModelMetric(value,missing,:perturbation)

    return metric

end

using CounterfactualExplanations
using CounterfactualExplanations.Models: probs
using CounterfactualExplanations.DataPreprocessing: unpack
using DataFrames
using StatsBase
"""
    mmd_model(experiment::Experiment, recourse_system::RecourseSystem; n=1000, grid_search=false, kwargs...)

Calculates the MMD on the probabilities of classification assigned by the model to the set of (all) instances. Allows to quantify the model shift.
"""
function mmd_model(experiment::Experiment, recourse_system::RecourseSystem; n=1000, grid_search=false, kwargs...)
    
    X, _ = unpack(experiment.data)

    if grid_search
        X = reduce(hcat,[map(x -> rand(range(x..., length=100)), extrema(X, dims=2)) for i in 1:n])
    end
    
    # Initial:
    M = recourse_system.initial_model
    proba = probs(M, X)

    # New:
    new_M = recourse_system.model
    new_proba = probs(new_M, X)

    value, p_value = mmd(proba, new_proba, 1000; compute_p=n, kwargs...)
    metric_name = grid_search ? :mmd_grid : :mmd

    metric = ModelMetric(value,p_value,metric_name)

    return metric

end

using LinearAlgebra
"""
    decisiveness(experiment::Experiment, recourse_system::RecourseSystem)

Calculates the pseudo-distance of points to the decision boundary measured as the average probability of classification centered around 0.5. High value corresponds to a large margin of classification.
"""
function decisiveness(experiment::Experiment, recourse_system::RecourseSystem)
    
    X, _ = unpack(experiment.data)

    # Initial:
    M = recourse_system.initial_model
    proba = probs(M, X)
    
    # New:
    new_M = recourse_system.model
    new_proba = probs(new_M, X)

    value = abs(norm(proba.-0.5) - norm(new_proba.-0.5))

    metric = ModelMetric(value,missing,:decisiveness)

    return metric
end

"""
    disagreement(experiment::Experiment, recourse_system::RecourseSystem)    

Calculates the Disagreement pseudo-distance defined in https://doi.org/10.1145/1273496.1273541 as Pr(h(x) != h'(x)), that is the probability that labels assigned by one classifier do not agree with the labels assigned by another classifier. Simply put, it measures the overlap between models. As this is an empirical measure, we can vary the number of records in `data`.
"""
function disagreement(experiment::Experiment, recourse_system::RecourseSystem)

    X, _ = unpack(experiment.data)

    # Initial:
    M = recourse_system.initial_model
    proba = reduce(hcat, map(x -> length(x) == 1 ? [x,1-x] : x, probs(M,X)))
    # New:
    new_M = recourse_system.model
    new_proba = reduce(hcat, map(x -> length(x) == 1 ? [x,1-x] : x, probs(new_M,X)))

    value = sum(argmax(proba,dims=1) .!= argmax(new_proba,dims=1))/size(X,2)
    metric = ModelMetric(value,missing,:disagreement)

    return metric
end

using MLJ, Flux
function fscore(experiment::Experiment, recourse_system::RecourseSystem)

    X, y = unpack(experiment.test_data)
    m = MulticlassFScore()
    binary = length(unique(y)) == 2

    # Initial:
    M = recourse_system.initial_model
    proba = reduce(hcat, map(x -> length(x) == 1 ? [x,1-x] : x, probs(M,X)))
    ŷ = Flux.onecold(proba, 1:size(proba,2))
    if binary
        ŷ .-= 1
    end
    fscore = m(ŷ, vec(y))

    # New:
    new_M = recourse_system.model
    new_proba = reduce(hcat, map(x -> length(x) == 1 ? [x,1-x] : x, probs(new_M,X)))
    new_ŷ = Flux.onecold(new_proba, 1:size(new_proba,2))
    if binary
        new_ŷ .-= 1
    end
    new_fscore = m(new_ŷ, vec(y))

    value = new_fscore - fscore    

    metric = ModelMetric(value,missing,:fscore)

    return metric
    
end


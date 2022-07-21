
using CounterfactualExplanations
using CounterfactualExplanations.Models: probs
using CounterfactualExplanations.DataPreprocessing: unpack
using DataFrames
using StatsBase
"""
    mmd_model(experiment::Experiment, recourse_system::RecourseSystem; n=1000, grid_search=false, kwargs...)

Calculates the MMD on the probabilities of classification assigned by the model to the set of (all)  instances. Allows to quantify the model shift.
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
    metric = mmd(proba, new_proba, n; kwargs...)
    
    metric_name = grid_search ? :mmd_grid : :mmd

    return metric_to_dataframe((metric...,:all), metric_name, :model)

end

using LinearAlgebra
"""
    decisiveness(experiment::Experiment, recourse_system::RecourseSystem)

Calculates the pseudo-distance of points to the decision boundary measured as the average probability of classification centered around 0.5. High value corresponds to a large margin of classification.
"""
function decisiveness(experiment::Experiment, recourse_system::RecourseSystem)
    
    X, y = unpack(experiment.data)
    # Initial:
    M = recourse_system.initial_model
    proba = probs(M, X)
    # New:
    new_M = recourse_system.model
    new_proba = probs(new_M, X)

    _classes = sort(unique(y))
    _classes = Int.(_classes)
    metric = map(cls -> (abs(norm(proba.-0.5) - norm(new_proba.-0.5)), missing, cls), _classes)

    return metric_to_dataframe(metric, :decisiveness, :model)
end

"""
    disagreement(experiment::Experiment, recourse_system::RecourseSystem)    

Calculates the Disagreement pseudo-distance defined in https://doi.org/10.1145/1273496.1273541 as Pr(h(x) != h'(x)), that is the probability that labels assigned by one classifier do not agree with the labels assigned by another classifier. Simply put, it measures the overlap between models. As this is an empirical measure, we can vary the number of records in `data`.
"""
function disagreement(experiment::Experiment, recourse_system::RecourseSystem)
    X, y = unpack(experiment.data)
    # Initial:
    M = recourse_system.initial_model
    proba = reduce(hcat, map(x -> length(x) == 1 ? [x,1-x] : x, probs(M,X)))
    # New:
    new_M = recourse_system.model
    new_proba = reduce(hcat, map(x -> length(x) == 1 ? [x,1-x] : x, probs(new_M,X)))

    _classes = sort(unique(y))
    _classes = Int.(_classes)
    metric = map(cls -> (sum(argmax(proba,dims=1) .!= argmax(new_proba,dims=1))/size(X,2), missing, cls), _classes)

    return metric_to_dataframe(metric, :disagreement, :model)
end



using CounterfactualExplanations
using CounterfactualExplanations.Models: probs
using CounterfactualExplanations.DataPreprocessing: unpack
using DataFrames
function mmd_model(experiment::Experiment, recourse_system::RecourseSystem, n=1000; target_only=true, as_dataframe=true, kwargs...)
    # Initial:
    X, y = unpack(experiment.data)
    M = recourse_system.initial_model
    proba = probs(M, X)
    # New:
    new_M = recourse_system.model
    new_proba = probs(new_M, X)
    _classes = target_only ? experiment.target : sort(unique(y))
    # Compute metric:
    metric = map(cls -> (mmd(proba[:,vec(y.==cls)], new_proba[:,vec(y.==cls)], n; kwargs...)..., cls), _classes)
    if as_dataframe
        if !isa(metric, AbstractVector)
            metric = [metric]
        end
        metric = DataFrame(NamedTuple{(:value, :p_value, :class)}.(metric))
        metric.metric .= :mmd
        metric.scope .= :model
    end
    return metric
end
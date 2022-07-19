using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: unpack
using DataFrames
function mmd_domain(experiment::Experiment, recourse_system::RecourseSystem, n=1000; target_only::Bool=true, as_dataframe=true, kwargs...)
    X, y = unpack(experiment.data)
    new_X, new_y = unpack(recourse_system.data)
    _classes = target_only ? experiment.target : sort(unique(y))
    metric = map(cls -> (mmd(X[:,vec(y.==cls)], new_X[:,vec(new_y.==cls)], n; kwargs...)..., cls), _classes)
    if as_dataframe
        if !isa(metric, AbstractVector)
            metric = [metric]
        end
        metric = DataFrame(NamedTuple{(:value, :p_value, :class)}.(metric))
        metric.metric .= :mmd
        metric.scope .= :domain
    end
    return metric
end
using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: unpack
using DataFrames

"""
    mmd_domain(experiment::Experiment, recourse_system::RecourseSystem, n=1000; target_only::Bool=true, kwargs...)

Calculates MMD for the input data by class.
"""
function mmd_domain(experiment::Experiment, recourse_system::RecourseSystem; n=1000, target_only::Bool=true, kwargs...)
    X, y = unpack(experiment.data)
    new_X, new_y = unpack(recourse_system.data)
    _classes = target_only ? experiment.target : sort(unique(y))
    _classes = Int.(_classes)
    metric = map(cls -> (mmd(X[:,vec(y.==cls)], new_X[:,vec(new_y.==cls)], n; kwargs...)..., cls), _classes)
    
    return metric_to_dataframe(metric, :mmd, :domain)
end




using DataFrames

struct DomainMetric <: AbstractMetric
    metric::Number
    p_value::Union{Missing,Number}
    name::Symbol
end

"""
    DataFrame(metric::DomainMetric)

Turns an instance of class `DomainMetric` into `DataFrame`.
"""
function DataFrames.DataFrame(metric::DomainMetric)
    vals = (metric.metric, metric.p_value, metric.name, :domain)
    df = DataFrame(NamedTuple{(:value, :p_value, :name, :scope)}.([vals]))
    return df
end


"""
    mmd_domain(experiment::Experiment, recourse_system::RecourseSystem, n=1000; target_only::Bool=true, kwargs...)

Calculates MMD for the input data.
"""
function mmd_domain(experiment::Experiment, recourse_system::RecourseSystem; n=1000, n_samples=1000, target_only::Bool=true, kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(experiment.data)
    new_X, new_y = CounterfactualExplanations.DataPreprocessing.unpack_data(recourse_system.data)
    if target_only
        value, p_value = mmd(X[:, vec(y .== experiment.target)], new_X[:, vec(new_y .== experiment.target)], n_samples; compute_p=n, kwargs...)
    else
        value, p_value = mmd(X, new_X, n_samples; compute_p=n, kwargs...)
    end

    metric = DomainMetric(value, p_value, :mmd)

    return metric
end




module Evaluation

using AlgorithmicRecourseDynamics
using CounterfactualExplanations
using Distances
using StatsBase
using LinearAlgebra
using ..Models
import ..Models: perturbation
using ..Experiments
using ..Experiments: Experiment, RecourseSystem
const MetricOrFun = Union{PreMetric,Function}

abstract type AbstractMetric end

include("kernels.jl")
include("mmd.jl")
include("domain_shifts.jl")
include("model_shifts.jl")

using DataFrames
function evaluate_system(
    recourse_system::RecourseSystem,
    experiment::Experiment;
    to_dataframe=true,
    n=1000,
    n_samples=1000,
)
    metrics = [
        mmd_domain(experiment, recourse_system; n=n),
        perturbation(experiment, recourse_system),
        mmd_model(experiment, recourse_system; n=n),
        mmd_model(experiment, recourse_system; n=n, grid_search=true, n_samples=n_samples),
        disagreement(experiment, recourse_system),
        decisiveness(experiment, recourse_system),
        model_performance(experiment, recourse_system),
    ]

    if to_dataframe
        metrics = reduce(vcat, map(DataFrame, metrics))
    end

    return metrics
end

export evaluate_system

end

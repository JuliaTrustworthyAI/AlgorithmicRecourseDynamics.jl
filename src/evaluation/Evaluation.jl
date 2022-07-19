module Evaluation

using Distances
using StatsBase
using LinearAlgebra
using ..Experiments
using ..Experiments: Experiment, RecourseSystem
const MetricOrFun = Union{PreMetric,Function}

include("kernels.jl")
include("mmd.jl")
include("domain_shifts.jl")
include("model_shifts.jl")

using DataFrames
function evaluate_system(recourse_system::RecourseSystem, experiment::Experiment)
    vcat(
        mmd_model(experiment, recourse_system, 1000),
        mmd_domain(experiment, recourse_system, 1000)
    )
end

end
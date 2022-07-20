module Evaluation

using Distances
using StatsBase
using LinearAlgebra
using ..Experiments
using ..Experiments: Experiment, RecourseSystem
const MetricOrFun = Union{PreMetric,Function}

include("utils.jl")
include("kernels.jl")
include("mmd.jl")
include("domain_shifts.jl")
include("model_shifts.jl")

using DataFrames
function evaluate_system(recourse_system::RecourseSystem, experiment::Experiment; n=1000)
    vcat(
        mmd_domain(experiment, recourse_system; n=n),
        mmd_model(experiment, recourse_system; n=n),
        mmd_model(experiment, recourse_system; n=n, grid_search=true),
        disagreement(experiment, recourse_system),
        decisiveness(experiment, recourse_system)
    )
end

end
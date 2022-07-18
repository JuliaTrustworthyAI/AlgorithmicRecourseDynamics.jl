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


end
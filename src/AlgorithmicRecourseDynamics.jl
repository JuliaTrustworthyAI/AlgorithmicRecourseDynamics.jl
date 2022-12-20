module AlgorithmicRecourseDynamics

using CounterfactualExplanations
using Logging

function is_logging(io)
    isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
end

# Load modules:
include("data/Data.jl")
using .Data
export load_synthetic, load_real_world

include("models/Models.jl")
using .Models

include("experiments/Experiments.jl")
using .Experiments

include("evaluation/Evaluation.jl")
using .Evaluation
export evaluate_system

include("base.jl")
export run_experiment, set_up_experiment, run_experiments, set_up_experiments, ExperimentResults

include("post_processing.jl")
export kable

end
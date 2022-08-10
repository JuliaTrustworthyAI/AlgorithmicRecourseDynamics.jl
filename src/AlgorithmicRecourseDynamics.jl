module AlgorithmicRecourseDynamics

# Load modules:
include("data/Data.jl")
using .Data
export load_synthetic, load_real_world

include("models/Models.jl")
using .Models

include("generators/Generators.jl")
using .Generators
export GravitationalGenerator, EndoROARGenerator

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
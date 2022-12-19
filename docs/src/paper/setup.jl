setup = quote
    # Environment:
    using Pkg; Pkg.activate("docs/src/paper")

    # Deps:
    using AlgorithmicRecourseDynamics
    using AlgorithmicRecourseDynamics.Models
    using AlgorithmicRecourseDynamics.Models: model_evaluation
    using CounterfactualExplanations
    using CounterfactualExplanations: counterfactual, counterfactual_label
    using DataFrames
    using Flux
    using Images
    using LaplaceRedux
    using Markdown
    using MLJBase
    using Plots
    using Random
    using Serialization

    # Setup
    Random.seed!(2023)              # global seed to allow for reproducibility
    theme(:wong)
    include("docs/src/utils.jl")    # some helper functions
    output_path = output_dir("poc") # output directory for artifacts
    www_path = www_dir("poc")       # output directory for images
end
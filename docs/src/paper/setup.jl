setup = quote
    # Environment:
    using Pkg
    Pkg.activate("docs/src/paper")

    # Deps:
    using AlgorithmicRecourseDynamics
    using AlgorithmicRecourseDynamics: run_bootstrap
    using AlgorithmicRecourseDynamics.Models
    using AlgorithmicRecourseDynamics.Models: model_evaluation
    using CounterfactualExplanations
    using CounterfactualExplanations: counterfactual, counterfactual_label
    using CSV
    using DataFrames
    using Flux
    using Images
    using LaplaceRedux
    using Markdown
    using MLJBase
    using MLJModels: OneHotEncoder
    using MLUtils
    using MLUtils: undersample
    using Plots
    using Random
    using RCall
    using Serialization
    using StatsBase

    # Setup
    Random.seed!(2023)              # global seed to allow for reproducibility
    theme(:wong)
    include("docs/src/utils.jl")    # some helper functions

    # Make DataFrames.jl work
    MLUtils.getobs(data::DataFrame, i) = data[i, :]
    MLUtils.numobs(data::DataFrame) = nrow(data)

end
setup_docs = quote
    using Pkg
    Pkg.activate("docs")

    using AlgorithmicRecourseDynamics
    using AlgorithmicRecourseDynamics.Data
    using AlgorithmicRecourseDynamics.Experiments
    using AlgorithmicRecourseDynamics.Models
    using AlgorithmicRecourseDynamics: run!
    using CounterfactualExplanations
    using Flux
    using MLJBase
    using Plots
    using Random

    Random.seed!(2023)
    theme(:wong)
end

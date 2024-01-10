import CompatHelperLocal as CHL
CHL.@check()

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
using Test

@testset "AlgorithmicRecourseDynamics.jl" begin
    N = 1000
    counterfactual_data = CounterfactualExplanations.load_linearly_separable(N)
    generator = GenericGenerator()

    data_train, data_test = CounterfactualExplanations.DataPreprocessing.train_test_split(
        counterfactual_data
    )
    mod = CounterfactualExplanations.fit_model(data_train, :MLP)

    models = Dict(:mymodel => mod)
    generators = Dict(:wachter => generator)
    experiment = set_up_experiment(data_train, data_test, models, generators)

    run!(experiment)
end

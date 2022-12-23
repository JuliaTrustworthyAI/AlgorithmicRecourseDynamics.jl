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
    xmax = 2
    X, ys = make_blobs(
        N, 2;
        centers=2, as_table=false, center_box=(-xmax => xmax), cluster_std=0.1
    )
    ys .= ys .== 2
    X = X'
    counterfactual_data = CounterfactualData(X, ys')

    n_epochs = 100
    model = Chain(Dense(2, 1))
    mod = FluxModel(model)
    generator = GenericGenerator()

    data_train, data_test = Data.train_test_split(counterfactual_data)
    Models.train(mod, data_train; n_epochs=n_epochs)

    models = Dict(:mymodel => mod)
    generators = Dict(:wachter => generator)
    experiment = set_up_experiment(data_train, data_test, models, generators)

    run!(experiment)

end

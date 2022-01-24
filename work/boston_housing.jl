include("../src/load.jl")
using AlgorithmicRecourse, MLDatasets, Flux
using Plots, PlotThemes
theme(:juno)
using Logging
disable_logging(Logging.Info)
output_folder = "output/boston_housing_ensemble"

using MLDatasets, Statistics
X = BostonHousing.features()
y = BostonHousing.targets()
y = Float64.(y .>= median(y)); # binary target

# Prepare data and model:
using Random
Random.seed!(1234)
using StatsBase
dt = fit(ZScoreTransform, X, dims=2)
StatsBase.transform!(dt, X)
xs = Flux.unstack(X,2)
data = zip(xs,y)
nn = Models.build_model(input_dim=size(X)[1], n_hidden=100)
loss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y)

run = false
if run
  # Train model:
  using Flux.Optimise: update!, ADAM
  using Statistics, StatsBase
  opt = ADAM()
  epochs = 100
  avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))
  accuracy(data) = sum(map(d ->round.(Flux.σ.(nn(d[1]))) .== d[2], data))[1]/length(data)

  using Plots
  anim = Animation()
  avg_l = [avg_loss(data)]
  p1 = scatter( ylim=(0,avg_l[1]), xlim=(0,epochs), legend=false, xlab="Epoch", title="Average loss")
  acc = [accuracy(data)]
  p2 = scatter( ylim=(0.5,1), xlim=(0,epochs), legend=false, xlab="Epoch", title="Accuracy")

  for epoch = 1:epochs
    for d in data
      gs = gradient(params(nn)) do
        l = loss(d...)
      end
      update!(opt, params(nn), gs)
    end
    avg_l = vcat(avg_l,avg_loss(data))
    plot!(p1, [0:epoch], avg_l, color=1)
    scatter!(p1, [0:epoch], avg_l, color=1)
    acc = vcat(acc,accuracy(data))
    plot!(p2, [0:epoch], acc, color=1)
    scatter!(p2, [0:epoch], acc, color=1)
    plt=plot(p1,p2, size=(600,300))
    frame(anim, plt)
  end

  gif(anim, "www/boston_housing_single_nn.gif", fps=10);
end

opt = ADAM()
loss_type = :logitbinarycrossentropy
run = false
if run
    K = 50
    𝓜 = Models.build_ensemble(K,kw=(input_dim=size(X)[1], n_hidden=100));
    𝓜, anim = Models.forward(𝓜, data, opt, n_epochs=30, plot_every=10, loss_type=loss_type); # fit the ensemble
    Models.save_ensemble(𝓜, root=output_folder) # save to disk
    gif(anim, "www/boston_housing_ensemble_loss.gif", fps=25);
end

# 𝓜 = Models.load_ensemble(root=output_folder)
# 𝑴 = Models.FittedEnsemble(𝓜,opt,loss_type);
# using Random
# Random.seed!(1234)
# x̅ = X[:,(y.==0)'][:,rand(1:length((y.==0)'))] # select individual sample
# x̅ = reshape(x̅, (length(x̅),1))
# γ = 0.75
# target=1.0
# T = 1000
# n = round(size(X)[2])
# δ = 0.1
# generator = GreedyGenerator(δ,n,:logitbinarycrossentropy,nothing)
# recourse = generate_recourse(generator, x̅, 𝑴, target, γ, T=T); # generate recourse

𝓜 = Models.load_ensemble(root=output_folder)
𝑴 = Models.FittedEnsemble(𝓜, opt, loss_type);
target=1.0
T = 1000
n = round(size(X)[2])
δ = 0.1
generator = GreedyGenerator(δ,n,:logitbinarycrossentropy,nothing)

# Variables:
μ = [0.01,0.05,0.1]
γ = [0.55,0.75,0.9]
grid_ = Experiments.GridVariables(μ, γ)
n_rounds = 2
# Experiment:
experiment = Experiments.Experiment(X,y,𝑴,target,grid_,n_rounds);

# retrain(𝑴,data)

# outcome = Experiments.run_experiment(experiment, generator, 1, T=T)
# outcome

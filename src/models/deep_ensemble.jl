using Flux, Plots

"""
    build_model(;input_dim=2,n_hidden=32,output_dim=1)

Helper function that builds a single neural network.
"""
function build_model(;input_dim=2,n_hidden=32,output_dim=1,p=0.3)
    
    nn = Chain(
        Dense(input_dim, n_hidden, relu),
        Dropout(p),
        Dense(n_hidden, n_hidden, relu),
        Dropout(p),
        Dense(n_hidden, output_dim))

    return nn

end


"""
    build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))

Helper function that builds an ensemble of `K` models.
"""
function build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))
    ensemble = [build_model(;kw...) for i in 1:K]
    return ensemble
end

using Flux.Optimise: update!
"""
    forward_nn(nn, loss, data, opt; n_epochs=200, plotting=nothing)

Trains a single neural network `nn`.
"""
function forward_nn(nn, loss, data, opt; n_epochs=200, plotting=nothing)

    avg_l = []
    
    for epoch = 1:n_epochs
      for d in data
        gs = gradient(Flux.params(nn)) do
          l = loss(d...)
        end
        update!(opt, Flux.params(nn), gs)
      end
      if !isnothing(plotting)
        plt = plotting[1]
        anim = plotting[2]
        idx = plotting[3]
        avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))
        avg_l = vcat(avg_l,avg_loss(data))
        x_range = maximum([epoch-plotting[4],1]):epoch
        if epoch % plotting[4]==0 
          plot!(plt, x_range, avg_l[x_range], color=idx, alpha=0.3)
          frame(anim, plt)
        end
      end
    end
    
end

using Statistics

"""
    forward(𝓜, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=200, plot_every=20) 

Trains a deep ensemble by separately training each neural network.
"""
function forward(𝓜, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=200, plot_every=20) 

    anim = nothing
    if plot_loss
        anim = Animation()
        plt = plot(ylim=(0,1), xlim=(0,n_epochs), legend=false, xlab="Epoch", title="Average (training) loss")
        for i in 1:length(𝓜)
            nn = 𝓜[i]
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=(plt, anim, i, plot_every))
        end
    else
        plt = nothing
        for nn in 𝓜
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=plt)
        end
    end

    return 𝓜, anim
end;

using BSON: @save
"""
    save_ensemble(𝓜::AbstractArray; root="")

Saves all models in ensemble to disk.
"""
function save_ensemble(𝓜::AbstractArray; root="")
    for i in 1:length(𝓜)
        path = root * "/nn" * string(i) * ".bson"
        model = 𝓜[i]
        @save path model
    end
end

using BSON: @load
"""
    load_ensemble(root="")

Loads all models in `root` folder and stores them in a list.
"""
function load_ensemble(;root="")
    all_files = Base.Filesystem.readdir(root)
    is_bson_file = map(file -> Base.Filesystem.splitext(file)[2][2:end], all_files) .== "bson"
    bson_files = all_files[is_bson_file]
    bson_files = map(file -> root * "/" * file, bson_files)
    𝓜 = []
    for file in bson_files
        @load file model
        𝓜 = vcat(𝓜, model)
    end
    return 𝓜
end

using AlgorithmicRecourse
import AlgorithmicRecourse.Models: logits, probs # import functions in order to extend

"""
    FittedEnsemble(𝓜::AbstractArray,opt::Any,loss_type::Symbol)

A simple subtype that is compatible with the AlgorithmicRecourse.jl package.
"""
struct FittedEnsemble <: AlgorithmicRecourse.Models.FittedModel
    𝓜::AbstractArray
    opt::Any
    loss_type::Symbol
end

"""
    logits(𝑴::FittedEnsemble, X::AbstractArray)

A method (extension) that computes predicted logits for a deep ensemble.
"""
logits(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.flatten(Flux.stack([nn(X) for nn in 𝑴.𝓜],1)),dims=1)

"""
    probs(𝑴::FittedEnsemble, X::AbstractArray)

A method (extension) that computes predicted probabilities for a deep ensemble.
"""
probs(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.flatten(Flux.stack([σ.(nn(X)) for nn in 𝑴.𝓜],1)),dims=1)

function retrain(𝑴::FittedEnsemble, data; n_epochs=200) 
    𝓜 = copy(𝑴.𝓜)
    𝓜 = forward(𝓜, data, 𝑴.opt, loss_type=𝑴.loss_type, plot_loss=false, n_epochs=n_epochs)
    𝑴 = FittedEnsemble(𝓜, 𝑴.opt, 𝑴.loss_type)
    return 𝑴
end


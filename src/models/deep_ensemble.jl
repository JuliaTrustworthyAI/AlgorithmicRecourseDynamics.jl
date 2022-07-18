using Flux, Plots

"""
    build_model(;input_dim=2,n_hidden=32,output_dim=1)

Helper function that builds a single neural network. If available, model is moved to GPU.
"""
function build_model(;input_dim=2,n_hidden=32,output_dim=1,p=0.3)
    
    nn = Chain(
        Dense(input_dim, n_hidden, relu),
        Dropout(p),
        Dense(n_hidden, n_hidden, relu),
        Dropout(p),
        Dense(n_hidden, output_dim)) |> gpu

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
function forward_nn(nn, loss, data, opt; n_epochs=200, plotting=nothing, τ=1.0)

    avg_l = []

    # Helper function for stopping criterium:
    accuracy() = sum(map(d ->round.(Flux.σ.(nn(d[1]))) .== d[2], data))[1]/length(data)
    stopping_criterium_reached = accuracy() >= τ
    epoch = 1

    while epoch <= n_epochs && !stopping_criterium_reached
        for d in data
            gs = gradient(Flux.params(nn)) do
                loss(d...)
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

        # Check if desired accuracy reached:
        stopping_criterium_reached = accuracy() >= τ
        epoch += 1
    end

    return nn
    
end

using CounterfactualExplanations
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

"""
    FittedNeuralNet(ensemble::AbstractArray,opt::Any,loss::Function)

A simple subtype that is compatible with the CounterfactualExplanations.jl package.
"""
struct FittedNeuralNet <: CounterfactualExplanations.AbstractFittedModel
    nn::Any
    opt::Any
    loss::Function
end

"""
    logits(M::FittedNeuralNet, X::AbstractArray)

A method (extension) that computes predicted logits for a single deep neural network.
"""
logits(M::FittedNeuralNet, X::AbstractArray) = M.nn(X)

"""
    probs(M::FittedNeuralNet, X::AbstractArray)

A method (extension) that computes predicted probabilities for a single deep neural network.
"""
probs(M::FittedNeuralNet, X::AbstractArray) = Flux.σ.(logits(M, X))

"""
    retrain(M::FittedNeuralNet, data; n_epochs=10)

Retrains a fitted a neural network for (new) data.
"""
function retrain(M::FittedNeuralNet, data; n_epochs=10, τ=1.0) 
    nn = M.nn
    nn = forward_nn(nn, M.loss, data, M.opt, n_epochs=n_epochs, τ=τ)
    M = FittedNeuralNet(nn, M.opt, M.loss)
    return M
end

using Statistics
"""
    forward(ensemble, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=200, plot_every=20) 

Trains a deep ensemble by separately training each neural network.
"""
function forward(ensemble, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=200, plot_every=20, τ=1.0) 

    anim = nothing
    if plot_loss
        anim = Animation()
        plt = plot(ylim=(0,1), xlim=(0,n_epochs), legend=false, xlab="Epoch", title="Average (training) loss")
        for i in 1:length(ensemble)
            nn = ensemble[i]
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            nn = forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=(plt, anim, i, plot_every), τ=τ)
            ensemble[i] = nn
        end
    else
        plt = nothing
        for i in 1:length(ensemble)
            nn = ensemble[i]
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            nn = forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=plt, τ=τ)
            ensemble[i] = nn
        end
    end

    return ensemble, anim
end;

using BSON: @save
"""
    save_ensemble(ensemble::AbstractArray; root="")

Saves all models in ensemble to disk.
"""
function save_ensemble(ensemble::AbstractArray; root="")
    for i in 1:length(ensemble)
        path = root * "/nn" * string(i) * ".bson"
        model = ensemble[i]
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
    ensemble = []
    for file in bson_files
        @load file model
        ensemble = vcat(ensemble, model)
    end
    return ensemble
end

using CounterfactualExplanations
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

"""
    FittedEnsemble(ensemble::AbstractArray,opt::Any,loss_type::Symbol)

A simple subtype that is compatible with the CounterfactualExplanations.jl package.
"""
struct FittedEnsemble <: CounterfactualExplanations.AbstractFittedModel
    ensemble::AbstractArray
    opt::Any
    loss_type::Symbol
end

"""
    logits(M::FittedEnsemble, X::AbstractArray)

A method (extension) that computes predicted logits for a deep ensemble.
"""
logits(M::FittedEnsemble, X::AbstractArray) = mean(Flux.flatten(Flux.stack([nn(X) for nn in M.ensemble],1)),dims=1)

"""
    probs(M::FittedEnsemble, X::AbstractArray)

A method (extension) that computes predicted probabilities for a deep ensemble.
"""
probs(M::FittedEnsemble, X::AbstractArray) = mean(Flux.flatten(Flux.stack([σ.(nn(X)) for nn in M.ensemble],1)),dims=1)

"""
    retrain(M::FittedEnsemble, data; n_epochs=10) 

Retrains a fitted deep ensemble for (new) data.
"""
function retrain(M::FittedEnsemble, data; n_epochs=10, τ=1.0) 
    ensemble = copy(M.ensemble)
    ensemble, anim = forward(ensemble, data, M.opt, loss_type=M.loss_type, plot_loss=false, n_epochs=n_epochs, τ=τ)
    M = FittedEnsemble(ensemble, M.opt, M.loss_type)
    return M
end


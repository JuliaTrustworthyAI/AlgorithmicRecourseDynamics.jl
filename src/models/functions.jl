"""
    prepare_data(X::AbstractArray, y::AbstractArray)

Helper function that prepares the data for training and moves arrays to GPU if available.
"""
function prepare_data(X::AbstractArray, y::AbstractArray)
    xs = Flux.unstack(X,2) |> gpu
    data = zip(xs,y)
    return data
end

"""
    build_model()

Helper function to build simple MLP.

# Examples

```julia-repl
using BayesLaplace
nn = build_model()
```

"""
function build_model(;input_dim=2,n_hidden=32,output_dim=1,batch_norm=false,dropout=false,activation=Flux.relu)
    
    if batch_norm
        nn = Chain(
            Dense(input_dim, n_hidden),
            BatchNorm(n_hidden, activation),
            Dense(n_hidden, output_dim),
            BatchNorm(output_dim)
        )  
    elseif dropout
        nn = Chain(
            Dense(input_dim, n_hidden, activation),
            Dropout(0.1),
            Dense(n_hidden, output_dim)
        )  
    else
        nn = Chain(
            Dense(input_dim, n_hidden, activation),
            Dense(n_hidden, output_dim)
        )  
    end

    return nn

end

using Flux
using Flux.Optimise: update!
"""
    forward_nn(nn, loss, data, opt; n_epochs=200, plotting=nothing)

Wrapper function to train neural network and generate an animation showing the training loss evolution.
"""
function forward_nn(nn, loss, data, opt; n_epochs=200, plotting=nothing)

    avg_l = []
    
    for epoch = 1:n_epochs
      for d in data
        gs = Flux.gradient(Flux.params(nn)) do
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
        if epoch % plotting[4]==0
          plot!(plt, avg_l, color=idx)
          frame(anim, plt)
        end
      end
    end
    
end

"""
    build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))

Helper function to build a simple ensemble composed of `K` MLPs.

# Examples

```julia-repl
using BayesLaplace
𝑬 = build_ensemble(5)
```

"""
function build_ensemble(K=5;kw=(input_dim=2,n_hidden=32,output_dim=1))
    ensemble = [build_model(;kw...) for i in 1:K]
    return ensemble
end

using Statistics
"""
    forward(ensemble, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=200, plot_every=20) 

Wrapper function to train deep ensemble and generate an animation showing the training loss evolution.
"""
function forward(ensemble, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=10, plot_every=1) 

    anim = nothing
    if plot_loss
        anim = Animation()
        plt = plot(ylim=(0,1), xlim=(0,n_epochs), legend=false, xlab="Epoch", title="Average (training) loss")
        for i in 1:length(ensemble)
            nn = ensemble[i]
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=(plt, anim, i, plot_every))
        end
    else
        plt = nothing
        for nn in ensemble
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=plt)
        end
    end

    return ensemble, anim
end
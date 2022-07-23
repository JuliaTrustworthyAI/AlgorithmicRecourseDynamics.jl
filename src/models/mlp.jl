using Parameters
@with_kw struct FluxModelParams 
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 10
    data_loader::Function = data_loader
end

using Flux
using Flux.Optimise: update!
using CounterfactualExplanations
"""
    train(M::FluxModel, data::CounterfactualData; kwargs...)

Wrapper function to retrain `FluxModel`.
"""
function train(M::FluxModel, data::CounterfactualData; τ=nothing, kwargs...)

    args = FluxModelParams(; kwargs...)

    # Prepare data:
    data = args.data_loader(data)

    # Training:
    model = M.model
    forward!(
        model, data; 
        loss = args.loss,
        opt = args.opt,
        n_epochs = args.n_epochs
    )

    return M
    
end

using Statistics
function forward!(model, data; loss::Symbol, opt::Symbol, n_epochs::Int=10)

    # Loss:
    loss_(x, y) = getfield(Flux.Losses, loss)(model(x), y) 
    avg_loss(data) = mean(map(d -> loss_(d[1],d[2]), data))

    # Optimizer:
    opt_ = getfield(Flux.Optimise, opt)()

    # Training:    
    for epoch = 1:n_epochs
        for d in data
            gs = Flux.gradient(Flux.params(model)) do
                l = loss_(d...)
            end
            update!(opt_, Flux.params(model), gs)
        end
    end

end

using MLUtils
"""
    data_loader(data::CounterfactualData)

Prepares data for training.
"""
function data_loader(data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    xs = MLUtils.unstack(X,dims=2) 
    data = zip(xs,y)
    return data
end

"""
    build_mlp()

Helper function to build simple MLP.

# Examples

```julia-repl
nn = build_mlp()
```

"""
function build_mlp(;input_dim=2,n_hidden=32,output_dim=1,batch_norm=false,dropout=false,activation=Flux.relu)
    
    if batch_norm
        model = Chain(
            Dense(input_dim, n_hidden),
            BatchNorm(n_hidden, activation),
            Dense(n_hidden, output_dim),
            BatchNorm(output_dim)
        )  
    elseif dropout
        model = Chain(
            Dense(input_dim, n_hidden, activation),
            Dropout(0.1),
            Dense(n_hidden, output_dim)
        )  
    else
        model = Chain(
            Dense(input_dim, n_hidden, activation),
            Dense(n_hidden, output_dim)
        )  
    end

    return model

end


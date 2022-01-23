"""
    Grid(μ::AbstractArray, γ::AbstractArray, ℓ::AbstractArray)

Sets up the the grid of variables. `μ` refers to the proportion of individuals that shall
recieve recourse and `γ` refers to the desired threshold probability for recourse.
"""
struct GridVariables
    μ::AbstractArray
    γ::AbstractArray
end

"""
    Grid(grid::Base.Iterators.ProductIterator)

The unfolded grid containing all combinations of all variables.
"""
struct Grid
    grid::Base.Iterators.ProductIterator
end

"""
    build_grid(grid::GridVariables) = Base.Iterators.product(grid.μ, grid.γ)    

Builds the grid based on provided variables.
"""
build_grid(grid::GridVariables) = Grid(Base.Iterators.product(grid.μ, grid.γ))


using AlgorithmicRecourse

"""
    Experiment(X::AbstractArray,y::AbstractArray,𝑴::AlgorithmicRecourse.Models.FittedModel,target::AbstractFloat,grid::Base.Iterators.ProductIterator,n_rounds::Int)

Sets up the experiment to be run.
"""
struct Experiment
    X::AbstractArray
    y::AbstractArray
    𝑴::AlgorithmicRecourse.Models.FittedModel
    target::AbstractFloat
    grid::GridVariables
    n_rounds::Int
end

using Random, StatsBase, LinearAlgebra, Flux
using ..Models

"""
    run_experiment(experiment::Experiment, generator::AlgorithmicRecourse.Generator, n_folds=5; seed=nothing, T=1000)

A wrapper function that runs the experiment for endogenous models shifts.
"""
function run_experiment(experiment::Experiment, generator::AlgorithmicRecourse.Generator, n_folds=5; seed=nothing, T=1000)

    # Setup:
    if !isnothing(seed)
        Random.seed!(seed)
    end
    output = []
    grid = build_grid(experiment.grid)

    for k in n_folds
        for (μ,γ) in grid.grid

            X = copy(experiment.X)
            y = copy(experiment.y)
            xs = Flux.unstack(X,2)
            data = zip(xs,y)
            chosen_individuals = []
            
            for t in 1:experiment.n_rounds

                # Classifier:
                if t > 1
                    𝑴 = Models.retrain(experiment.𝑴, data)
                end
    
                # Choose individuals:
                adverse_outcome = findall(vec(experiment.y .!=  experiment.target))
                n_individuals = Int(round(μ * length(adverse_outcome)))
                chosen_individualsₜ = StatsBase.sample(adverse_outcome,n_individuals,replace=false)

                # Generate recourse:
                for i in chosen_individualsₜ
                    x̅ = X[:,i]
                    recourse = generate_recourse(generator, x̅, experiment.𝑴, experiment.target, γ, T=T)
                    X[:,i] = recourse.x̲ # update individuals features
                end

                # Evaluate recourse:
                chosen_individuals = union(chosen_individuals, chosen_individualsₜ)
                pct_validₜ = sum(round.(AlgorithmicRecourse.Models.probs(experiment.𝑴, X[:,chosen_individuals])) .== experiment.target)/length(chosen_individuals)
                ΔX = X[:,chosen_individuals] - experiment.X[:,chosen_individuals]
                avg_costₜ = norm(ΔX, 2)

                # Collect and store output:
                outputₜ = (
                    pct_valid=pct_validₜ, 
                    avg_cost=avg_costₜ,
                    t = t,
                    μ = μ,
                    γ = γ,
                    k = k
                )
                output = vcat(output, outputₜ)
                
            end
        end
    end

    return output

end


#### Old stuff




# function run_experiment(X,y,model,target,generators,generator_args,experiment;scale=false)

#     # Setup:
#     if scale
#         X, dt = scale(X)
#     end

#     # Run:
#     evaluation = DataFrame()
#     recourse_path = DataFrame()
#     for i in 1:length(generators)
#         generator_name = string(keys(generators)[i])
#         evaluation_i, recourse_path_i, clf_path_i = experiment(X,y,model,target,generators[i],generator_args[i])
#         insertcols!(evaluation_i, :generator => generator_name)
#         evaluation = vcat(evaluation, evaluation_i)
#     end
#     return evaluation
# end;

# # -------------- Holdout:
# function experiment_holdout(X,y,classifier,target,generator,generator_args;gradient=gradient,proportion_training=0.5,proportion_holdout=1.0,n_folds=5)
#     # Setup:
#     N = length(y);
#     D = size(X)[2]; # number of features
#     train_indices, test_indices = kfolds_(N;k=n_folds)
#     avg_cost = zeros(0)
#     pct_valid = zeros(0)
#     recourse_path = DataFrame()
#     clf_path = []
    
#     for k in 1:n_folds
#         # 0. Fold setup:
#         train_rows = train_indices[k]
#         test_rows = test_indices[k]
#         X_train, y_train = (X[train_rows,:], y[train_rows])
#         X_test, y_test = (X[test_rows,:], y[test_rows])
        
#         # 1. Train (M1):
#         M1 = classifier(X_train,y_train);

#         # 2. Recourse on training:
#         examples = findall(y_train.!=target);
#         eligible_examples = StatsBase.sample(examples,Int(round(proportion_training * length(examples))),replace=false);
#         X_train_cf = copy(X_train)
#         y_train_cf = copy(y_train)
#         for i in eligible_examples
#             x_f = X_train_cf[i,:]
#             recourse = generator(x_f,gradient,M1,1;generator_args...)
#             X_train_cf[i,:] = recourse.x_cf 
#             y_train_cf[i] = recourse.y_cf
#         end

#         # 3. Retrain (M2):
#         M2 = classifier(X_train_cf,y_train_cf);

#         # 4. Recourse for holdout:
#         examples = findall(y_test.==0);
#         eligible_examples = StatsBase.sample(examples,Int(round(proportion_holdout * length(examples))),replace=false);
#         X_test_cf = copy(X_test)
#         y_test_cf = copy(y_test)
#         avg_cost_k = 0
#         pct_valid_k = 0
#         N_eligible = length(eligible_examples)
#         for i in eligible_examples
#             # Implement recourse
#             x_f = X_test_cf[i,:]
#             recourse = generator(x_f,gradient,M1,1;generator_args...) # recourse against M1
#             X_test_cf[i,:] = recourse.x_cf 
#             y_test_cf[i] = predict(M2, vcat(1, recourse.x_cf); proba=false)[1] # label according to M2
#             # Validity and cost
#             avg_cost_k += recourse.cost/N_eligible
#             pct_valid_k += valid(recourse; classifier=M2)/N_eligible
#         end
        
#         # 5. Collect output:
#         pct_valid_k = round(pct_valid_k; digits=5);
#         avg_cost = vcat(avg_cost,avg_cost_k)
#         pct_valid = vcat(pct_valid,pct_valid_k)
#     end
    
#     # Output:
#     evaluation = DataFrame(hcat(pct_valid, avg_cost),[:validity, :cost])
#     insertcols!(evaluation, :period => 1)
#     insertcols!(evaluation, :fold => 1:n_folds)
    
#     recourse_path = nothing
#     clf_path = nothing
    
#     return evaluation, recourse_path, clf_path
# end;

# # --------------- Dynamic: 
# function experiment_dynamic(X,y,classifier,target,generator,generator_args;gradient=gradient,proportion_recourse=0.01,n_folds=10,n_rounds=10)
#     # Setup:
#     N = length(y);
#     D = size(X)[2] # number of features
#     evaluation = DataFrame()
    
#     # CV:
#     for k in 1:n_folds
        
#         # Allocating memory:
#         global recourse_path = DataFrame()
#         global clf_path = []
#         results = DataFrame(idx=1:N, received_recourse=false, label=y, validity=1.0, cost=0.0)
#         X_train = copy(X)
#         y_train = copy(y)
#         avg_cost = zeros(0)
#         pct_valid = zeros(0)

#         # Recursion:
#         t = 1
#         while t<=n_rounds

#             # Train classifier:
#             model = classifier(X_train,y_train) # might want to use w_t-1 as new prior
#             clf_path = vcat(clf_path, model)
#             w = model.μ
#             # w_0 = w # posterior as new prior
#             # H_0 = model.Σ
            
#             # Provide recourse:
#             adverse_outcome = findall(y_train.!=target)
#             N_0 = length(adverse_outcome)
#             chosen_individualsₜ = StatsBase.sample(adverse_outcome,Int(round(proportion_recourse * N_0)),replace=false)
#             for i in chosen_individualsₜ
#                 # Implement recourse:
#                 x_f = X_train[i,:]
#                 recourse = generator(x_f,gradient,model,target;generator_args...)
#                 X_train[i,:] = recourse.x_cf
#                 idx = findall(results.idx .== i)[1]
#                 results[idx,:received_recourse] = true
#                 results[idx,:cost] = recourse.cost
#             end
#             y_train = predict(model, X_train)
#             results.label .= y_train
#             results.validity .= results.label .== target 
#             recourse_outcome = DataFrame(hcat(y_train,X_train), vcat("y", "x" .* string.(1:D)))
#             insertcols!(recourse_outcome, :period => t)
#             recourse_path = vcat(recourse_path, recourse_outcome)
            
#             # Validity and cost
#             avg_cost_t = mean(results[(results.received_recourse),:].cost)
#             pct_valid_t = mean(results[(results.received_recourse),:].validity)
#             avg_cost = vcat(avg_cost,avg_cost_t)
#             pct_valid = vcat(pct_valid,pct_valid_t)
            
#             # Updates:
#             t += 1
#         end

#         # Output:
#         evaluation_k = DataFrame(hcat(pct_valid, avg_cost),[:validity, :cost])
#         insertcols!(evaluation_k, :period => 1:n_rounds)
#         insertcols!(evaluation_k, :fold => k)
#         evaluation = vcat(evaluation, evaluation_k)

#     end
    
#     return evaluation, recourse_path, clf_path
# end;


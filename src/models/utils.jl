# Train-test split
using StatsBase
using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: unpack
using Flux
"""
    train_test_split(data::CounterfactualData;test_size=0.2)

Splits data into train and test split.
"""
function train_test_split(data::CounterfactualData;test_size=0.2)
    X,y = unpack(data)
    N = size(y,2)
    classes_ = sort(unique(y))
    n_per_class = round(N/length(classes_))
    test_idx = sort(reduce(vcat,[sample(findall(vec(y.==cls)), Int(floor(test_size * n_per_class)),replace=false) for cls in classes_]))
    train_idx = setdiff(1:N, test_idx)
    train_data = CounterfactualData(X[:,train_idx], y[:,train_idx])
    test_data = CounterfactualData(X[:,test_idx], y[:,test_idx])
    return train_data, test_data
end


function undersample(data::CounterfactualData, n_per_class::Int)
    
    X,y = unpack(data)
    n_classes, N = size(y)
    if n_classes > 2
        y_cls = Flux.onecold(y,1:n_classes)
    else
        y_cls = y
    end
    classes_ = sort(unique(y_cls))

    idx = sort(reduce(vcat,[sample(findall(vec(y_cls.==cls)), n_per_class,replace=false) for cls in classes_]))
    data = CounterfactualData(X[:,idx], y[:,idx])

    return data

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

using CounterfactualExplanations
using MLJ, Flux
function model_evaluation(M::CounterfactualExplanations.AbstractFittedModel, test_data::CounterfactualData)
    X, y = unpack(test_data)
    m = MulticlassFScore()
    binary = M.likelihood == :classification_binary
    if binary
        proba = reduce(hcat, map(x -> binary ? [1-x,x] : x, probs(M,X)))
        ŷ = Flux.onecold(proba, 0:1)
    else
        y = Flux.onecold(y,1:size(y,1))
        ŷ = Flux.onecold(probs(M,X), sort(unique(y)))
    end
    fscore = m(ŷ, vec(y))

    return fscore
end
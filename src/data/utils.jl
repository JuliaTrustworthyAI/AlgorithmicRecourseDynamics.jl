using CounterfactualExplanations.DataPreprocessing: CounterfactualData
using DataFrames
using Flux
using StatsBase

function Base.hcat(data::CounterfactualData, more_data::CounterfactualData)

    data = deepcopy(data)
    more_data = deepcopy(more_data)

    @assert all(data.features_categorical .== more_data.features_categorical) "Datasets have different categorical indices."
    @assert all(data.features_continuous .== more_data.features_continuous) "Datasets have different continous indices."

    data.X = hcat(data.X, more_data.X)
    data.y = hcat(data.y, more_data.y)

    return data
end

function DataFrames.subset(data::CounterfactualData, idx::Vector{Int})
    dsub = deepcopy(data)
    dsub.X = dsub.X[:,idx]
    dsub.y = dsub.y[:,idx]
    return dsub
end

"""
    train_test_split(data::CounterfactualData;test_size=0.2)

Splits data into train and test split.
"""
function train_test_split(data::CounterfactualData;test_size=0.2)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    N = size(y,2)
    classes_ = sort(unique(y))
    n_per_class = round(N/length(classes_))
    test_idx = sort(reduce(vcat,[sample(findall(vec(y.==cls)), Int(floor(test_size * n_per_class)),replace=false) for cls in classes_]))
    train_idx = setdiff(1:N, test_idx)
    train_data = subset(data, train_idx)
    test_data = subset(data, test_idx)
    return train_data, test_data
end

function undersample(data::CounterfactualData, n_per_class::Int)
    
    X,y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    n_classes, N = size(y)
    if n_classes > 2
        y_cls = Flux.onecold(y,1:n_classes)
    else
        y_cls = y
    end
    classes_ = sort(unique(y_cls))

    idx = sort(reduce(vcat,[sample(findall(vec(y_cls.==cls)), n_per_class,replace=false) for cls in classes_]))
    data.X = X[:, idx]
    data.y = y[:,idx]

    return data

end
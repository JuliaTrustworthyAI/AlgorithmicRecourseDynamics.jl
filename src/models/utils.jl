# Train-test split
using StatsBase
using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: unpack
function train_test_split(data::CounterfactualData;test_size=0.2,balanced=true)
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
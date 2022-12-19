using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing
using Flux
using MLJBase
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

function model_evaluation(M::CounterfactualExplanations.AbstractFittedModel, test_data::CounterfactualData)
    X, y = DataPreprocessing.unpack(test_data)
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
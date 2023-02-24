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
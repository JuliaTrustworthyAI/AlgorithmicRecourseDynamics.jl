using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: unpack
function mmd_domain(experiment::Experiment; target_only::Bool=true)
    X, y = unpack(experiment.data)
    new_X, new_y = unpack(experiment.newdata)
    _classes = target_only ? experiment.target : sort(unique(y))
    return map(cls -> (mmd(X[:,y.==cls], new_X[:,new_y.==cls])..., cls), _classes)
end
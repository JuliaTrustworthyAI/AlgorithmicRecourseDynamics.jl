
using CounterfactualExplanations
using CounterfactualExplanations.Models: probs
using CounterfactualExplanations.DataPreprocessing: unpack
function mmd(experiment::Experiment;target_only=true)
    # Initial:
    X, y = unpack(experiment.data)
    M = experiment.model
    proba = probs(M, X)
    # New:
    new_M = experimen.newmodel
    new_proba = probs(new_M, X)
    _classes = target_only ? experiment.target : sort(unique(y))
    return map(cls -> (mmd(proba[:,y.==cls], new_proba[:,new_y.==cls])..., cls), _classes)
end
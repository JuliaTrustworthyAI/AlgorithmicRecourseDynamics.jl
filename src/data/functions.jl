using Pkg.Artifacts

function download_artifacts()
    data_dir = artifact"data"
end

using StatsBase
function scale(X, dim)
    dt = fit(ZScoreTransform, X, dim=dim)
    X_scaled = StatsBase.transform(dt, X)
    return X_scaled, dt
end

function rescale(X, dt)
    X_rescaled = StatsBase.reconstruct(dt, X)
    return X_rescaled
end
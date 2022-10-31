using LazyArtifacts
using CounterfactualExplanations
using CSV
using DataFrames
using StatsBase

function load_synthetic(max_obs::Union{Nothing, Int}=nothing)
    data_dir = joinpath(artifact"data","data/synthetic")
    files = readdir(data_dir)
    files = files[contains.(files,".csv")]
    data = map(files) do file
        df = CSV.read(joinpath(data_dir, file), DataFrame)
        X = convert(Matrix, hcat(df.x1,df.x2)')
        y = convert(Matrix, df.target')
        data = CounterfactualData(X,y)
        if !isnothing(max_obs)
            n_classes = length(unique(y))
            data = undersample(data, Int(round(max_obs/n_classes)))
        end
        (Symbol(replace(file, ".csv" => "")) => data)
    end
    data = Dict(data...)
    return data
end

function load_real_world(max_obs::Union{Nothing, Int}=nothing)
    data_dir = joinpath(artifact"data","data/real_world")
    files = readdir(data_dir)
    files = files[contains.(files,".csv")]
    data = map(files) do file
        df = CSV.read(joinpath(data_dir, file), DataFrame)
        X = Matrix(df[:,Not(:target)])
        X = permutedims(X)
        y = convert(Matrix, df.target')
        data = CounterfactualData(X,y)
        if !isnothing(max_obs)
            n_classes = length(unique(y))
            data = undersample(data, Int(round(max_obs/n_classes)))
        end
        (Symbol(replace(file, ".csv" => "")) => data)
    end
    data = Dict(data...)
    return data
end


function scale(X, dim)
    dt = fit(ZScoreTransform, X, dim=dim)
    X_scaled = StatsBase.transform(dt, X)
    return X_scaled, dt
end

function rescale(X, dt)
    X_rescaled = StatsBase.reconstruct(dt, X)
    return X_rescaled
end
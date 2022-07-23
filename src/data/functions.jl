using Pkg.Artifacts

using CSV, DataFrames, CounterfactualExplanations
function load_synthetic()
    data_dir = joinpath(artifact"data","data/synthetic")
    files = readdir(data_dir)
    data = map(files) do file
        df = CSV.read(joinpath(data_dir, file), DataFrame)
        X = convert(Matrix, hcat(df.feature1,df.feature2)')
        y = convert(Matrix, df.target')
        data = CounterfactualData(X,y)
        (Symbol(replace(file, ".csv" => "")) => data)
    end
    data = Dict(data...)
    return data
end

function load_real_world()
    data_dir = joinpath(artifact"data","data/real_world")
    files = readdir(data_dir)
    data = map(files) do file
        df = CSV.read(joinpath(data_dir, file), DataFrame)
        X = convert(Matrix, hcat(df.feature1,df.feature2)')
        y = convert(Matrix, df.target')
        data = CounterfactualData(X,y)
        (Symbol(replace(file, ".csv" => "")) => data)
    end
    data = Dict(data...)
    return data
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
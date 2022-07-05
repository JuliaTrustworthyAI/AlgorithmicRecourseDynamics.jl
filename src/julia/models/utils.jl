"""
    prepare_data(X::AbstractArray, y::AbstractArray)

Helper function that prepares the data for training and moves arrays to GPU if available.
"""
function prepare_data(X::AbstractArray, y::AbstractArray)
    xs = Flux.unstack(X,2) |> gpu
    data = zip(xs,y)
    return data
end
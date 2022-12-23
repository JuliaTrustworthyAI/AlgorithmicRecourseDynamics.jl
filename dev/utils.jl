"""
    output_dir(dir="")

Sets up the directory to save computational outputs and returns the path.
"""
function output_dir(dir="")
    root_ = "dev/artifacts/upload/output"
    output_dir = joinpath(root_, dir)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    return output_dir
end

"""
    www_dir(dir="")

Sets up the directory to save images and returns the path.
"""
function www_dir(dir="")
    root_ = "dev/artifacts/upload/www"
    www_dir = joinpath(root_, dir)
    if !isdir(www_dir)
        mkpath(www_dir)
    end
    return www_dir
end

"""
    data_dir(dir="")

Sets up the directory to save images and returns the path.
"""
function data_dir(dir="")
    root_ = "dev/artifacts/upload/data"
    data_dir = joinpath(root_, dir)
    if !isdir(data_dir)
        mkpath(data_dir)
    end
    return data_dir
end

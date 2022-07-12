"""
    output_dir(dir="")

Sets up the directory to save computational outputs and returns the path.
"""
function output_dir(dir="")
    root_ = "dev/artifacts/upload/output"
    if !isdir(root_)
        mkpath(root_)
    end
    output_dir = joinpath(root_, dir)
    return output_dir
end

"""
    www_dir(dir="")

Sets up the directory to save images and returns the path.
"""
function www_dir(dir="")
    root_ = "dev/artifacts/upload/www"
    if !isdir(root_)
        mkpath(root_)
    end
    www_dir = joinpath(root_, dir)
    return www_dir
end


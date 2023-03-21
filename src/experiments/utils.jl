# Compute cartesian product over two vectors:
function expandgrid(x, y)
    N = length(x) * length(y)
    grid = (z -> Matrix(transpose(reshape(z, (2, N)))))(
        collect(Iterators.flatten(Iterators.product(x, y)))
    )
    return grid
end

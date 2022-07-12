# Compute cartesian product over two vectors:
function expandgrid(x,y)
    N = length(x) * length(y)
    grid = Iterators.product(x,y) |>
        Iterators.flatten |>
        collect |>
        z -> reshape(z, (2,N)) |>
        transpose |>
        Matrix
    return grid
end

# Train-test split
using StatsBase
function train_test_split(X,y;test_size=0.2)
    N = size(X)[1]
    rows = 1:N
    test_rows = StatsBase.sample(rows,Int(floor(test_size * N)),replace=false)
    train_rows = setdiff(rows, test_rows)
    X_train = X[train_rows,:]
    y_train = y[train_rows]
    X_test = X[test_rows,:]
    y_test = y[test_rows]
    return X_train, y_train, X_test, y_test
end

# K-fold
using MLDataUtils
function kfolds_(N;k=5)
    rows = 1:N
    rows_shuffled = StatsBase.sample(rows,N,replace=false)
    folds = kfolds(rows_shuffled, k)
    train_indices = [folds.data[idx] for idx in folds.train_indices]
    test_indices = [folds.data[idx] for idx in folds.val_indices]
    return train_indices, test_indices
end
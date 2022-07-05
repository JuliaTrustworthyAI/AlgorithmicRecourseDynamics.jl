import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(dataset, model, output_directory, generator_name,
                      plot_type, plot_id, show_plot=False):
    """
    Generates a plot a two-class dataset and saves it to a file.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model (MLModelCatalog):
            Classifier implementing a `predict_proba()` method.
        output_directory (str):
            Name of the directory where images are saved.
        generator_name (str):
            Name of the applied recourse generator.
        plot_type (str):
            Type of the created plot.
        plot_id (str):
            ID for the generated plot (e.g. consecutive numbers for different distributions).
        show_plot (Boolean):
            If True the plot will also be outputted directly to the notebook.
    """
    # Plot only two dimensional data
    if dataset._df.to_numpy().shape[1] != 3:
        return

    train = dataset._df_train.to_numpy()
    test = dataset._df_test.to_numpy()
    fig, ax = plt.subplots()

    fig.set_dpi(300)
    ax.set_aspect('equal')
    ax.set_xlim([-0.50, 1.50])
    ax.set_ylim([-0.25, 1.25])
    ax.set_xlabel('$feature1$')
    ax.set_ylabel('$feature2$')

    x0, x1, z = calculate_boundary(dataset._df, model)
    ax.contourf(x0, x1, z, cmap='plasma', levels=10, alpha=0.8)

    y = test[:, 2]
    # Plot test samples
    ax.scatter(test[y == 1, 0], test[y == 1, 1], s=30, c='darkorange',
               linewidth=0.6, edgecolor='black', marker='X')
    ax.scatter(test[y == 0, 0], test[y == 0, 1], s=30, c='cornflowerblue',
               linewidth=0.6, edgecolor='black', marker='X')

    y = train[:, 2]
    # Plot train samples
    ax.scatter(train[y == 1, 0], train[y == 1, 1], s=30, c='darkorange',
               linewidth=0.6, edgecolor='black')
    ax.scatter(train[y == 0, 0], train[y == 0, 1], s=30, c='cornflowerblue',
               linewidth=0.6, edgecolor='black')

    # Save (and output) the figure
    figure = plt.gcf()
    if show_plot:
        plt.show()

    figure.savefig(f"{output_directory}/{generator_name}_{plot_type}_{f'{plot_id:06}'}.png",
                   bbox_inches='tight', dpi=300)
    plt.close()


def calculate_boundary(data, model, resolution=0.01, plot=True):
    """
    Calculates the 2D decision boundary for a contour plot.

    Args:
        data (pandas.DataFrame):
            Records along with their labels.
        model (MLModelCatalog):
            Classifier implementing a `predict_proba()` method.
        resolution (float, optional):
            1 / N for the number of points sampled in a unit length; defaults to 0.01.

    Returns:
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray): Meshgrid and predicted probabilities.
    """
    data = data.to_numpy()

    x_min = np.min(data[:, :], axis=0) - 1
    x_max = np.max(data[:, :], axis=0) + 1

    ranges = [np.arange(x_min[index], x_max[index], resolution) for index in range((data.shape[1] - 1))]
    grid = np.meshgrid(*ranges)

    x_new = grid[0].flatten().reshape((-1, 1))
    for i in range(1, len(grid)):
        x_new = np.c_[x_new, grid[i].flatten().reshape((-1, 1))]

    y_new = model.predict_proba(x_new)[:, 1]

    if plot:
        z = y_new.reshape(grid[0].shape)
        return grid[0], grid[1], z
    else:
        z = y_new.flatten().reshape((-1, 1))
        return x_new, z

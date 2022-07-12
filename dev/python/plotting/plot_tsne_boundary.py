import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE


def plot_tsne_boundary(data, target, model, title, file_name, bounds=None):
    """Visualize the multi-dimensional decision boundary using a simplified
    version of the technique described in https://link.springer.com/article/10.1007/s10618-013-0342-x

    Args:
        df (pandas.DataFrame):
            Dataframe containing all samples of the dataset.
        target (str):
            Name of the column that contains the target variable.
        model (MLModelCatalog):
            Classifier with additional utilities required by CARLA.
        title (str):
            Name of the plot, this should include the generator and epoch.
        file_name (str):
            Name of the file where the plot will be saved.
        bounds (dict of numpy.ndarray, optional):
            A dictionary containing describing the x- and y-axis bounds of the plot.

    Returns:
        dict of numpy.ndarray: A dictionary containing describing the x- and y-axis bounds of the plot.
    """
    X = data.loc[:, data.columns != target]
    y = data.loc[:, target]

    random.seed(42)
    np.random.seed(42)

    # tSNE is non-parametric so it is impossible to reuse it before and after recourse
    # Other dimensionality reduction or embedding techniques could be used.
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    y_predicted = np.argmax(model.predict_proba(data.loc[:, data.columns != target]), axis=1)

    resolution = 1000
    if bounds is not None:
        x_min = bounds['x_min']
        x_max = bounds['x_max']
    else:
        x_min = np.min(X_embedded[:, :], axis=0) * 1.25
        x_max = np.max(X_embedded[:, :], axis=0) * 1.25

    xx, yy = np.meshgrid(np.linspace(x_min[0], x_max[0], resolution),
                         np.linspace(x_min[1], x_max[1], resolution))

    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded, y_predicted)
    voronoi = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoi = voronoi.reshape((resolution, resolution))

    plt.contourf(xx, yy, voronoi, cmap='viridis', alpha=0.75)
    positive = X_embedded[y == 1]
    negative = X_embedded[y == 0]

    pos = plt.scatter(positive[:, 0], positive[:, 1], s=10, edgecolors='black',
                      linewidths=0.5, marker='o', c='darkorange')
    neg = plt.scatter(negative[:, 0], negative[:, 1], s=10, edgecolors='black',
                      linewidths=0.5, marker='v', c='cornflowerblue')

    plt.legend((pos, neg), ('Positive', 'Negative'), scatterpoints=1, loc=1)
    plt.suptitle(title)
    plt.xlabel('$feature1$')
    plt.ylabel('$feature2$')
    plt.colorbar()
    plt.savefig(file_name, bbox_inches='tight', dpi=500)
    plt.close()

    return {'x_min': x_min, 'x_max': x_max}

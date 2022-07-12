import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_continuous_dataset(means0, covs0, sizes0, means1, covs1, sizes1, file_name="data.csv"):
    """
    Generates a two-class dataset of continuous features where both classes
    contain a specified number of peaks drawn from Gaussian distributions.

    Args:
        means0 (list of numpy.ndarray):
            Means of all peaks in the distribution of the negative class.
        covs0 (list of numpy.ndarray):
            Covariance matrices for all peaks in the distribution of the negative class.
        sizes0 (list of int):
            Sizes (in number of samples) for all peaks in the distribution of the negative class.
        means1 (list of numpy.ndarray):
            Means of all peaks in the distribution of the positive class.
        covs1 (list of numpy.ndarray):
            Covariance matrices for all peaks in the distribution of the positive class.
        sizes1 (list of int):
            Sizes (in number of samples) for all peaks in the distribution of the positive class.
        file_name (str):
            Name of the file where the resulting dataset will be stored.

    Returns:
        numpy.ndarray: Dataset containing samples of two classes.
    """
    # Generate the random samples
    samples0 = np.random.multivariate_normal(means0[0], covs0[0], sizes0[0])
    samples1 = np.random.multivariate_normal(means1[0], covs1[0], sizes1[0])

    for i, _ in enumerate(sizes0[1:]):
        new_samples = np.random.multivariate_normal(means0[i + 1], covs0[i + 1], sizes0[i + 1])
        samples0 = np.r_[samples0, new_samples]

    for i, _ in enumerate(sizes1[1:]):
        new_samples = np.random.multivariate_normal(means1[i + 1], covs1[i + 1], sizes1[i + 1])
        samples1 = np.r_[samples1, new_samples]

    # Append labels to the classes
    class0 = np.c_[samples0, np.zeros(len(samples0), dtype=np.int8)]
    class1 = np.c_[samples1, np.ones(len(samples1), dtype=np.int8)]
    colors = np.array(['cornflowerblue', 'darkorange'])

    # Construct the dataset
    dataset = np.r_[class0, class1]

    # Plot the resulting distribution only if it contains two features + target
    if dataset.shape[1] == 3:
        plt.xlabel('$feature1$')
        plt.ylabel('$feature2$')
        plt.scatter(dataset[:, 0], dataset[:, 1], s=50, edgecolors='black',
                    linewidths=0.6, c=colors[dataset[:, 2].astype(int)])
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # Store in a csv file
    dataframe = pd.DataFrame(data=dataset, columns=['feature1', 'feature2', 'target'])
    dataframe.to_csv(file_name, index=False, float_format='%1.4f')
    return dataset


def generate_categorical_samples(size, ranges):
    """
    Generates a set of samples with normally-distributed categorical features.

    Args:
        size (int):
            Total number of samples to be generated.
        ranges (list of (int, int)):
            List containing the minimum and maximum value for each of the features.

    Returns:
        numpy.ndarray: Set of samples with categorical features.
    """
    # Initialize the numpy array representing all samples in the class
    samples = np.zeros((size, len(ranges)), dtype=np.int)

    # Iterate through all features
    for i in range(len(ranges)):
        mean = ranges[i][0] + (ranges[i][1] - ranges[i][0]) / 2
        # We want to ensure that effectively all observations are in the provided range
        std = (ranges[i][1] - mean) / 3.5

        # Generate a distribution for this feature
        distribution = np.random.normal(mean, std, size)
        for j in range(len(distribution)):
            # Clamp to the given range
            value = sorted((ranges[i][0], int(distribution[j]), ranges[i][1]))[1]
            samples[j, i] = value

    return samples


def generate_categorical_dataset(size0, ranges0, size1, ranges1, file_name="data.csv"):
    """
    Generates a two-class dataset of categorical features where both classes
    contain a specified number of features created based on Gaussian distributions.

    Args:
        size0 (int):
            Size (in number of samples) of the negative class.
        ranges0 (list of (int, int)):
            Tuples describing the minimum and maximum value for each feature for the negative class.
        size1 (int):
            Size (in number of samples) of the positive class.
        ranges1 (list of (int, int)):
            Tuples describing the minimum and maximum value for each feature for the positive class.
        file_name (str):
            Name of the file where the resulting dataset will be stored.

    Returns:
        numpy.ndarray: Dataset containing samples of two classes.
    """

    samples0 = generate_categorical_samples(size0, ranges0)
    samples1 = generate_categorical_samples(size1, ranges1)

    # Append labels to the classes
    class0 = np.c_[samples0, np.zeros(len(samples0), dtype=np.int)]
    class1 = np.c_[samples1, np.ones(len(samples1), dtype=np.int)]
    colors = np.array(['cornflowerblue', 'darkorange'])

    # Construct the dataset
    dataset = np.r_[class0, class1]

    # Plot the resulting distribution
    if dataset.shape[1] == 3:
        plt.scatter(dataset[:, 0], dataset[:, 1], s=50, edgecolors='black',
                    linewidths=0.6, c=colors[dataset[:, 2].astype(int)])
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # Store in a csv file
    columns = [f'feature{index + 1}' for index in range(len(ranges0))]
    columns.append('target')
    dataframe = pd.DataFrame(data=dataset, columns=columns)
    dataframe.to_csv(file_name, index=False, float_format='%1.4f')
    return dataset

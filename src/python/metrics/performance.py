import numpy as np


from sklearn.metrics import accuracy_score, f1_score


# TODO: Reconsider if this should be calculated on full dataset or only the test set
def measure_performance(dataset, model):
    """
    Computes a set of performance metrics for a classifier.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model (MLModelCatalog)
            Classifier with additional utilities required by CARLA.

    Returns:
        dict: A dictionary of statistical measurements of the model performance.
    """
    predictions = model.predict_proba(dataset.df_test.loc[:, dataset.df_test.columns != dataset.target])
    predicted_labels = np.argmax(predictions, axis=1)
    ground_truth = dataset.df_test.loc[:, dataset.df_test.columns == dataset.target].values.astype(int).flatten()

    return {
        "acc": accuracy_score(ground_truth, predicted_labels),
        "f1": f1_score(ground_truth, predicted_labels)
    }

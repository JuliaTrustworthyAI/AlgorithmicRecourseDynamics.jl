import carla.models.catalog.load_model as loading_utils
import carla.models.catalog.train_model as training_utils

from carla import log, MLModelCatalog
from carla.data.catalog import DataCatalog


class DynamicMLModelCatalog(MLModelCatalog):
    """
    Wrapper class for the MLModelCatalog that introduces additional functions
    allowing for the efficient and unbiased measurement of the dynamics of recourse.

    Attributes:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model_type (str):
            Black-box model used for classification, currently this class supports only ANNs and Logistic Regression.
        backend (str):
            Framework used on the backend, currently this class supports only PyTorch.
        save_name: (str):
            Name of the file with the model.
    """
    def __init__(self, data: DataCatalog, model_type: str, backend: str = "pytorch",
                 load_online: bool = False, **kwargs) -> None:

        if backend != 'pytorch':
            raise NotImplementedError("Only PyTorch models are currently supported")

        if model_type not in ['ann', 'linear']:
            raise NotImplementedError(f"Model type not supported: {self.model_type}")

        save_name = model_type
        if model_type == 'ann':
            save_name += f"_layers_{kwargs['save_name_params']}"
        self._save_name = save_name

        super().__init__(data=data, model_type=model_type, backend=backend,
                         load_online=load_online, **kwargs)

    @property
    def save_name(self) -> str:
        """
        Getter for the name of the file where the model is stored.

        Returns:
            str: Name of the file with the model.
        """
        return self._save_name

    def retrain(self, learning_rate=0.01, epochs=5, batch_size=1):
        """
        Loads a cached model and retrains it on an updated dataset.

        Args:
            learning_rate (float):
                Size of the step at each epoch of the model training.
            epochs (int):
                Number of iterations of training.
            batch_size (int):
                Number of samples used at once in a gradient descent step, if '1' the procedure is stochastic.
        """

        # Attempt to load the saved model
        self._model = loading_utils.load_trained_model(save_name=self._save_name,
                                                       data_name=self.data.name,
                                                       backend=self.backend)

        # This method should only be used when a model is already available
        if self._model is None:
            raise ValueError(f"No trained model found for {self._save_name}")

        # Sanity check to see if loaded model accuracy makes sense
        if self._model is not None:
            self._test_accuracy()

        # Get preprocessed data
        df_train = self.data.df_train
        df_test = self.data.df_test

        # All dataframes may have possibly changed
        x_train = df_train[list(set(df_train.columns) - {self.data.target})]
        y_train = df_train[self.data.target]
        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        # Order data (column-wise) before training
        x_train = self.get_ordered_features(x_train)
        x_test = self.get_ordered_features(x_test)

        log.info(f"Current balance: train set {y_train.mean()}, test set {y_test.mean()}")

        # Access the data in a format expected by PyTorch
        train_dataset = training_utils.DataFrameDataset(x_train, y_train)
        train_loader = training_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = training_utils.DataFrameDataset(x_test, y_test)
        test_loader = training_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Retrain the model
        training_utils._training_torch(self._model, train_loader, test_loader,
                                       learning_rate, epochs)

        loading_utils.save_model(model=self._model, save_name=self._save_name,
                                 data_name=self.data.name, backend=self.backend)

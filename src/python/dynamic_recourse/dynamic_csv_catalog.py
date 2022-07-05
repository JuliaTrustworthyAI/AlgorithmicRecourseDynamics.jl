import pandas as pd


from carla import log
from carla.data.catalog import DataCatalog
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from carla.data.pipelining import (decode,
                                   descale,
                                   encode,
                                   fit_encoder,
                                   fit_scaler,
                                   scale)
from typing import Callable, List, Tuple


class DynamicCsvCatalog(DataCatalog):
    """
    Wrapper class for the DataCatalog similar to the built-in CsvCatalog
    but with new capabilities required to control data in the experiments.

    Attributes:
        file_path (str):
            Path to the .csv file containing the dataset.
        categorical (List[str]):
            Names of columns describing categorical features.
        continuous (List[str]):
            Names of columns describing continuous (i.e. numerical) features.
        immutables (List[str]):
            Names of columns describing immutable features, not supported by all generators.
        target (str):
            Name of the column that contains the target variable.
        positive (int):
            Encoding of the positive class in the dataset (i.e. the goal of recourse).
        negative (int):
            Encoding of the negative class in the dataset (i.e. the target of recourse).
        name (str):
            Name assigned to this dataset, this is always "custom".
        scaler (BaseEstimator):
            Transformation that should be applied to scale continuous variables.
        encoder (BaseEstimator):
            Transformation that should be applied to encode categorical variables.
        identity_encoding (Boolean):
            True if the categorical features should not be encoded for this dataset.
        pipeline (List[Tuple[str, Callable]]):
            Set of operations bringing the dataset to its pre-processed form.
        inverse_pipeline (List[Tuple[str, Callable]]):
            Set of operations bringing the dataset to its original form.
        df (pandas.DataFrame):
            Dataframe containing all samples of the dataset.
        df_train (pandas.DataFrame):
            Part of the `df` that contains only the training set samples.
        df_test (pandas.DataFrame):
            Part of the `df` that contains only the test set samples.
    """
    def __init__(self, file_path: str, categorical: List[str],  continuous: List[str],
                 immutables: List[str], target: str, test_size: float = 0.5,
                 scaling_method: str = "MinMax", encoding_method: str = "OneHot_drop_binary",
                 positive: int = 1, negative: int = 0, name='custom'):

        self._categorical = categorical
        self._continuous = continuous
        self._immutables = immutables
        self._target = target
        self._positive = positive
        self._negative = negative
        self.name = name

        # Load the raw data
        raw = pd.read_csv(file_path)
        train_raw, test_raw = train_test_split(raw, test_size=test_size, stratify=raw[target])
        log.info(f"Balance: train set {train_raw[self.target].mean()}, test set {test_raw[self.target].mean()}")

        # Fit scaler and encoder
        if len(self.continuous) == 0:
            self.scaler: BaseEstimator = None
        else:
            self.scaler: BaseEstimator = fit_scaler(scaling_method, raw[self.continuous])

        self.encoder: BaseEstimator = fit_encoder(encoding_method, raw[self.categorical])

        self._identity_encoding = (encoding_method is None or encoding_method == "Identity")

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self._df = self.transform(raw)
        self._df_train = self.transform(train_raw)
        self._df_test = self.transform(test_raw)

    @property
    def categorical(self) -> List[str]:
        """
        Getter for the names of the categorical columns.

        Returns:
            List[str]: Names of the columns containing categorical features.
        """
        return self._categorical

    @property
    def continuous(self) -> List[str]:
        """
        Getter for the names of the continuous columns.

        Returns:
            List[str]: Names of the columns containing continuous features.
        """
        return self._continuous

    @property
    def immutables(self) -> List[str]:
        """
        Getter for the names of the immutable columns.

        Returns:
            List[str]: Names of the columns containing immutable features.
        """
        return self._immutables

    @property
    def target(self) -> str:
        """
        Getter for the name of the target column.

        Returns:
            str: Name of the column containing ground truth.
        """
        return self._target

    @property
    def positive(self) -> int:
        """
        Getter for the encoding of the positive class.

        Returns:
            int: Value in the target column encoding samples of the positive class.
        """
        return self._positive

    @property
    def negative(self) -> int:
        """
        Getter for the encoding of the negative class

        Returns:
            int: Value in the target column encoding samples of the negative class.
        """
        return self._negative

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        """
        Creates a pipeline to transform features of this dataset.

        Returns:
            List[Tuple[str, Callable]]: Set of operations bringing the dataset to its pre-processed form.
        """
        result = []
        if self.scaler is not None:
            result.append(("scaler", lambda x: scale(self.scaler, self.continuous, x)))
        if self.encoder is not None:
            result.append(("encoder", lambda x: encode(self.encoder, self.categorical, x)))
        return result

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        """
        Creates a pipeline to inverse the transformation of features of this dataset.

        Returns:
            List[Tuple[str, Callable]]: Set of operations bringing the dataset to its original form.
        """
        result = []
        if self.encoder is not None:
            result.append(("encoder", lambda x: decode(self.encoder, self.categorical, x)))
        if self.scaler is not None:
            result.append(("scaler", lambda x: descale(self.scaler, self.continuous, x)))
        return result

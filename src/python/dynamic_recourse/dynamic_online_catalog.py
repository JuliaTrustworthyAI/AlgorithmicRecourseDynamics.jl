from carla.data.catalog import OnlineCatalog


class DynamicOnlineCatalog(OnlineCatalog):
    """
    Wrapper class for the OnlineCatalog with new capabilities required to control data in the experiments.

    Attributes:
        target (str):
            Name of the column that contains the target variable.
        positive (int):
            Encoding of the positive class in the dataset (i.e. the goal of recourse).
        negative (int):
            Encoding of the negative class in the dataset (i.e. the target of recourse).
    """

    def __init__(self, data_name, target, positive=1, negative=0,
                 scaling_method: str = "MinMax", encoding_method: str = "OneHot_drop_binary"):

        self._target = target
        self._positive = positive
        self._negative = negative
        super().__init__(data_name, scaling_method, encoding_method)

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

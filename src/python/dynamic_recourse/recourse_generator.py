import os
import pandas as pd
import sys


from carla import log
from .dynamic_mlmodel_catalog import DynamicMLModelCatalog
from func_timeout import func_timeout, FunctionTimedOut
from inspect import signature


class RecourseGenerator():
    """
    Wrapper class for the CARLA generators that contains utilities required to conduct experiments.

    Attributes:
        name (str):
            Name of the generator.
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model (MLModelCatalog):
            Classifier with additional utilities required by CARLA.
        recourse_method (RecourseMethod):
            A generator of algorithmic recourse which follows the API of CARLA.
        generator_params (dict):
            Dictionary of parameters that affect the actions of the generator.
        model_params (dict):
            Hyper-parameters for the underlying black-box model.
        timeout (int):
            Number of seconds after which the generation of a counterfactual should be considered a failure.
    """
    def __init__(self, name, dataset, model, recourse_method,
                 generator_params, model_params, timeout=None):
        self.name = name
        self.dataset = dataset
        self.model = model
        self.recourse_method = recourse_method
        self.generator_params = generator_params
        self.model_params = model_params
        self.timeout = timeout
        self.positive_class_proba = 0

        self.update_generator()

    def update_generator(self):
        """
        Re-creates the generator on an updated model.
        """
        sig = signature(self.recourse_method.__init__)
        params = [param for param in sig.parameters]

        # RecourseMethods of CARLA do not follow a common interface
        if 'data' in params:
            self.generator = self.recourse_method(mlmodel=self.model,
                                                  data=self.dataset,
                                                  hyperparams=self.generator_params)
        else:
            self.generator = self.recourse_method(mlmodel=self.model,
                                                  hyperparams=self.generator_params)

    def apply(self, factuals):
        """
        Generate (a set of) counterfactuals with a method relevant for the generator.

        Args:
            factuals (pandas.DataFrame):
                One or more records from a dataset used to train the black-box model.

        Returns:
            int: Number of successfully generated counterfactuals.
        """
        if self.timeout is not None:
            counterfactuals = self.apply_recourse_with_timeout(factuals)
        else:
            counterfactuals = self.apply_recourse(factuals)
        self.dataset._df.update(counterfactuals)
        self.dataset._df_train.update(counterfactuals)
        return counterfactuals

    def apply_recourse(self, factuals):
        """
        Generate (a set of) counterfactual explanations with a provided generator.

        Args:
            factuals (pandas.DataFrame):
                One or more records from a dataset used to train the black-box model.

        Returns:
            int: Number of successfully generated counterfactuals.
        """
        log.info(f"Applying the {self.name} generator.")

        if factuals is None or len(factuals) == 0:
            return None

        return self.generator.get_counterfactuals(factuals).dropna()

    def apply_recourse_with_timeout(self, factuals):
        """
        Generate (a set of) counterfactual explanations with a provided generator.
        These explanations are applied one-by-one with a specific timeout for every single factual.

        Args:
            factuals (pandas.DataFrame):
                One or more records from a dataset used to train the black-box model.

        Returns:
            int: Number of successfully generated counterfactuals.
        """
        log.info(f"Applying the {self.name} generator.")

        if factuals is None or len(factuals) == 0:
            return None

        found_counterfactuals = None
        for i in range(len(factuals)):
            f = factuals.iloc[[i]]
            log.info(f"Generating counterfactual {i + 1} with {self.name}")
            # CARLA does not implement a timeout for the generators by default
            # but we want to prevent the code from running indefinitely
            counterfactual = recourse_controller(recourse_worker, self.timeout, self.generator, f)
            # We only want to overwrite the existing data if counterfactual generation was successful
            if counterfactual is not None and not counterfactual.empty:
                # Reset to the correct index
                counterfactual.rename(index={counterfactual.index[0]: f.index[0]}, inplace=True)
                if found_counterfactuals is None:
                    found_counterfactuals = counterfactual
                else:
                    found_counterfactuals = pd.concat([found_counterfactuals, counterfactual], axis=0)
        return found_counterfactuals

    def update_model(self):
        """
        Re-train the model based on an updated dataset.
        """

        # Ensure that the dataset saved by the model is always updated with counterfactuals
        self.model.data._df.update(self.dataset._df)
        self.model.data._df_train.update(self.dataset._df)
        self.model.data._df_test.update(self.dataset._df)

        log.info(f'Updating the {self.name} model')
        self.model = train_model(self.dataset, self.model_params, self.model.model_type, retrain=True)

    def describe(self):
        """
        Returns a dictionary containing basic information about this generator.

        Returns:
            dict: Dictionary that describes the generator including its type and hyper-parameters.
        """
        return {
            'type': self.recourse_method.__name__,
            'params': self.generator_params,
            'timeout': self.timeout
        }


def recourse_worker(generator, factual):
    """
    Apply algorithmic recourse for a (set of) factuals using a chosen generator.

    Args:
        generator (RecourseMethod):
            Generator that finds counterfactual explanations using a black-box model.
        factual (pandas.DataFrame):
            One or more records from a dataset used to train the black-box model.

    Returns:
        pandas.DataFrame: A counterfactual explanation for the provided factual.
    """
    if factual is None:
        raise ValueError('Provided with a non-existent factual')

    counterfactuals = generator.get_counterfactuals(factual).dropna()
    if not counterfactuals.empty:
        return counterfactuals.sample().astype(float)
    raise FunctionTimedOut()


def recourse_controller(function, max_wait_time, generator, factual):
    """
    Wrapper function that ensures the application of recourse does not run indefinitely.

    Args:
        function (Callable):
            Function that will have its execution placed under a timeout.
        max_wait_time (int):
            Number of seconds after which the `function` will time out.
        generator (RecourseMethod):
            Generator that finds counterfactual explanations using a black-box model.
        factual (pandas.DataFrame):
            One or more records from a dataset used to train the black-box model.

    Returns:
        pandas.DataFrame: A counterfactual explanation for the provided factual if found.
    """
    try:
        return func_timeout(max_wait_time, function, args=[generator, factual])
    except FunctionTimedOut:
        log.info("Timeout - No Counterfactual Explanation Found")

    return None


def train_model(dataset, hyper_params, model_type, retrain=False):
    """
    Instantiates and trains a black-box model within a CARLA wrapper that will be used to generate explanations.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        hyper_params (dict):
            Dictionary storing all custom hyper-parameter values for the model.
        model_type (str):
            Type of a model (ann / linear) that is supported by CARLA.
        retrain (Boolean):
            If true, the model is loaded from the file and retrained.

    Returns:
        MLModelCatalog: Classifier with additional utilities required by CARLA.
    """
    kwargs = {}
    if model_type == 'ann':
        kwargs = {'save_name_params': "_".join([str(size) for size in hyper_params['hidden_size']])}
    model = DynamicMLModelCatalog(dataset, model_type=model_type,
                                  load_online=False, backend="pytorch", **kwargs)

    # force_train is enabled to ensure that the model is not reused from the cache
    if not retrain:
        log.info("Training the initial model")
        block_print()
        model.train(learning_rate=hyper_params['learning_rate'],
                    epochs=hyper_params['epochs'],
                    batch_size=hyper_params['batch_size'],
                    hidden_size=hyper_params['hidden_size'],
                    force_train=True)
        enable_print()
    else:
        log.info("Retraining the model")
        block_print()
        model.retrain(learning_rate=hyper_params['learning_rate'],
                      epochs=hyper_params['epochs'],
                      batch_size=hyper_params['batch_size'])
        enable_print()

    return model


def block_print():
    """
    Reroute stdout to disable printing.
    """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """
    Activate printing again.
    """
    sys.stdout = sys.__stdout__

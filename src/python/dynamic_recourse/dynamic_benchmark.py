import numpy as np
import pandas as pd
import timeit


from carla import MLModelCatalog
from carla.evaluation.benchmark import Benchmark
from ..metrics.measurement import measure
from ..plotting.plot_dataset import plot_distribution


class DynamicBenchmark(Benchmark):
    """
    Wrapper around the Benchmark class which is used to gather data about the dynamics
    of a generator and to assess the quality of recourse at the end of the experiment.
    Supports different recourse methods.

    Attributes:
        mlmodel (MLModelCatalog):
            Classifier with additional utilities required by CARLA.
        generator (RecourseGenerator):
            An algorithmic recourse generator assessed in the experiment.
        recourse_method (RecourseMethod):
            A generator of algorithmic recourse which follows the API of CARLA.
        factuals (pandas.DataFrame):
            Factual instances which were provided to the generator during the experiment.
        counterfactuals (pandas.DataFrame):
            Counterfactual instances which were created by the generator.
        epoch (int):
            Current epoch in the experiment.
        timer (float):
            Number of fractional seconds taken to generate the counterfactuals.
    """
    def __init__(self, mlmodel, generator, recourse_method):
        self._mlmodel = mlmodel
        self._generator = generator
        self._recourse_method = recourse_method
        self._factuals = None
        self._counterfactuals = None
        self._epoch = 0
        self._timer = 0
        self._positive_class_proba = None

        # Avoid using scaling and normalizing more than once
        if isinstance(mlmodel, MLModelCatalog):
            self._mlmodel.use_pipeline = False

    def start(self, experiment_data, path, initial_measurements, calculate_p):
        """Executes the initial steps (epoch 0) of the experiment.

        Args:
            experiment_data (dict):
                Dictionary storing all data related to the experiment.
            path (str):
                Name of the directory where images are saved.
            calculate_p (Boolean):
                If True, the statistical significance is calculated for MMD of distribution and model.
        """
        experiment_data[self._generator.name][0] = measure(self._generator,
                                                           initial_measurements,
                                                           calculate_p)

        # Plot initial data distributions
        plot_distribution(self._generator.dataset, self._generator.model, path,
                          self._generator.name, 'distribution', self._epoch)

        # Due to some bug in CARLA a model needs to be retrained once before experiment begins
        self._generator.update_model()
        self._generator.update_generator()

    def next_iteration(self, experiment_data, path, current_factuals_index,
                       initial_measurements, calculate_p):
        """
        Executes an iteration of the experiment that consists of the generation of recourse,
        measurement of shifts, and update of the underlying model.

        Args:
            experiment_data (dict):
                Dictionary storing all data related to the ongoing experiment.
            path (str):
                Name of the directory where images are saved.
            current_factuals_index (list of int):
                Indexes of the counterfactuals which should be treated by all generators in this iteration.
            calculate_p (Boolean):
                If True, the statistical significance is calculated for MMD of distribution and model.
        """
        experiment_data[self._generator.name][self._epoch + 1] = {}

        # Find relevant factuals
        current_factuals = self._generator.dataset._df.iloc[current_factuals_index]

        if self._factuals is None:
            self._factuals = current_factuals
        else:
            self._factuals = pd.concat([self._factuals, current_factuals], axis=0)

        # Apply recourse
        start_time = timeit.default_timer()
        counterfactuals = self._generator.apply(current_factuals)
        if self._counterfactuals is None:
            self._counterfactuals = counterfactuals
        else:
            self._counterfactuals = pd.concat([self._counterfactuals, counterfactuals], axis=0)

        self._timer += timeit.default_timer() - start_time

        # Store the probabilities assigned by the model to counterfactuals
        if counterfactuals is not None and self._positive_class_proba is None:
            self._positive_class_proba = self._generator.model.predict_proba(
                counterfactuals.loc[:, counterfactuals.columns != self._generator.dataset.target]
                )[:, 1]
        elif counterfactuals is not None and self._positive_class_proba is not None:
            self._positive_class_proba = np.r_[
                self._positive_class_proba,
                self._generator.model.predict_proba(
                    counterfactuals.loc[:, counterfactuals.columns != self._generator.dataset.target]
                )[:, 1]
                ]

        # Measure the data distribution and performance of the model
        experiment_data[self._generator.name][self._epoch + 1] = measure(self._generator,
                                                                         initial_measurements,
                                                                         calculate_p)

        # Plot data distributions
        plot_distribution(self._generator.dataset, self._generator.model, path,
                          self._generator.name, 'distribution', self._epoch + 1)

        # Retrain the model on the updated dataset
        self._generator.update_model()

        # Re-create the generator on new model
        self._generator.update_generator()
        self._epoch += 1

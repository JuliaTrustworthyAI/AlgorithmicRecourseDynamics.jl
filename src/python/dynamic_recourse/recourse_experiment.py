import json
import numpy as np
import os


from carla import log
from carla.models.negative_instances import predict_negative_instances
from copy import deepcopy
from .dynamic_benchmark import DynamicBenchmark
from ..metrics.measurement import measure
from ..plotting.plot_dataset import calculate_boundary
from ..plotting.plot_tsne_boundary import plot_tsne_boundary
from datetime import datetime


class RecourseExperiment():
    """
    Allows to conduct an experiment about the dynamics of algorithmic recourse.

    Attributes:
        experiment_name (str):
            Name of the experiment with a distinguishing timestamp, used as the directory name where results are saved.
        initial_dataset (DataCatalog):
            Copy of the dataset before the implementation of recourse.
        initial_model (MLModelCatalog):
            Copy of the classifier before the implementation of recourse.
        initial_proba (numpy.ndarray):
            Predicted probabilities assigned to samples before the implementation of recourse.
        generators (List[RecourseGenerator]):
            List of one or more generators which will be measured in the experiment.
        hyper_params (dict):
            Dictionary of hyper-parameters provided to the models.
        initial_samples (dict of numpy.ndarray):
            Samples from the positive and negative class before the implementation of recourse.
        experiment_data (dict):
            Dictionary storing all data related to the ongoing experiment.
        benchmarks (List[Benchmark]):
            List of one or more Benchmark objects used to track the dynamics of the generators.
    """
    def __init__(self, dataset, model, generators, experiment_name='experiment', hyper_params={}):
        assert len(generators) != 0

        # Experiment data is saved into a new directory
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.experiment_name = f'{timestamp}_{experiment_name}'
        os.makedirs(f'../experiment_data/{self.experiment_name}')

        # Store the initial dataset and model used as baseline for the calculation of dynamics
        grid = None
        if dataset._df.to_numpy().shape[1] <= 3:
            grid, _ = calculate_boundary(dataset._df, model, resolution=0.1, plot=False)

        # Sample the initial classes for the calculations of MMD
        pos_individuals = dataset.df_train.loc[dataset.df_train[dataset.target] == dataset.positive]
        pos_sample = pos_individuals.sample(n=min(len(pos_individuals), 10000)).to_numpy()
        neg_individuals = dataset.df_train.loc[dataset.df_train[dataset.target] == dataset.negative]
        neg_sample = neg_individuals.sample(n=min(len(neg_individuals), 10000)).to_numpy()

        self.initial_measurements = {
            'dataset': deepcopy(dataset._df),
            'model': deepcopy(model),
            'grid': grid,
            'proba': model.predict_proba(dataset._df.loc[:, dataset._df.columns != dataset.target]),
            'samples': {
                'positive': pos_sample,
                'negative': neg_sample
            }
        }

        self.generators = generators
        self.hyper_params = hyper_params

        # Initialize the object storing all data about the experiment
        self.experiment_data = {}
        self.experiment_data['parameters'] = self.describe()

        self.benchmarks = {}
        for g in self.generators:
            self.experiment_data[g.name] = {0: {}}
            self.benchmarks[g.name] = DynamicBenchmark(model, g, g.recourse_method)

        file_name = f'../experiment_data/{self.experiment_name}/initial_boundary_tsne.png'
        title = f'Initial model on {dataset.name}'
        self.bounds = plot_tsne_boundary(dataset._df, dataset._target, model, title, file_name)

    def run(self, epochs=0.8, recourse_per_epoch=0.05, calculate_p=None):
        """
        Driver code to execute an experiment that allows to compare the dynamics of recourse
        applied by some generator to a benchmark described by Wachter et al. (2017).

        Attributes:
            total_recourse (float):
                Value between 0 and 1 representing the proportion of samples from the training set
                which should have recourse applied to them throughout the experiment.
            recourse_per_epoch (float):
                Value between 0 and 1 representing the proportion of samples from the training set
                which should have recourse applied to them in a single iteration.
            calculate_p (Boolean):
                If True, the statistical significance is calculated for MMD of distribution and model.
        """
        path = f'../experiment_data/{self.experiment_name}'

        # These parameters can be provided as floats referring to the proportion of the dataset.
        if isinstance(recourse_per_epoch, float):
            # Convert ratio of samples that should undergo recourse in a single epoch into a number
            recourse_per_epoch = max(int(recourse_per_epoch * len(self.factuals)), 1)
        if isinstance(epochs, float):
            # Convert ratio of samples that should undergo recourse in total into a number of epochs
            epochs = max(int(min(epochs, 1) * len(self.factuals) / recourse_per_epoch), 1)

        self.experiment_data['parameters']['epochs'] = epochs
        self.experiment_data['parameters']['recourse_per_epoch'] = recourse_per_epoch

        # Execute the initial measurements for all generators
        for g in self.generators:
            self.benchmarks[g.name].start(self.experiment_data, path, self.initial_measurements, calculate_p)

        # Apply recourse over the specified number of epochs
        for epoch in range(epochs):
            log.info(f"Starting epoch {epoch + 1}")

            # Get all samples from the original dataset
            current_neg_instances = self.initial_measurements['dataset'].index.to_list()
            for g in self.generators:
                # Leave in the list only those, which are predicted negative by all generators
                generator_factuals = predict_negative_instances(g.model, g.dataset.df_train).index.to_list()
                current_neg_instances = [f for f in current_neg_instances if f in generator_factuals]
            # Generate a subset S of factuals that have never been encountered by the generators
            sample_size = min(recourse_per_epoch, len(current_neg_instances))
            current_factuals_index = np.random.choice(current_neg_instances, replace=False, size=sample_size)
            # For sanity we remove duplicates (there should be none but CARLA seems to be buggy)
            current_factuals_index = np.unique(current_factuals_index)

            if len(current_factuals_index) == 0:
                log.info("No common negative instances found")
                for g in self.generators:
                    self.experiment_data[g.name][epoch + 1] = measure(g, self.initial_measurements, calculate_p)
                    # Plot the final decision boundary
                    file_name = f'../experiment_data/{self.experiment_name}/{g.name}_boundary_tsne.png'
                    title = f'{g.name} model on {g.dataset.name}. Round {epoch + 1}.'
                    plot_tsne_boundary(g.dataset._df, g.dataset._target, g.model, title, file_name, self.bounds)
                break

            p = calculate_p if (epoch + 1) == epochs else None

            # Apply the same set of actions on all generators passed to the experiment
            for g in self.generators:
                if (epoch + 1) % 2 == 0:
                    file_name = f'../experiment_data/{self.experiment_name}/{g.name}_boundary_{epoch + 1}_tsne.png'
                    title = f'{g.name} model on {g.dataset.name}. Round {epoch + 1}.'
                    plot_tsne_boundary(g.dataset._df, g.dataset._target, g.model, title, file_name, self.bounds)

                self.benchmarks[g.name].next_iteration(self.experiment_data, path,
                                                       current_factuals_index,
                                                       self.initial_measurements, p)

        # Measure the quality of recourse using CARLA tools
        self.experiment_data['evaluation'] = {}
        for g in self.generators:

            # Plot the final decision boundary
            file_name = f'../experiment_data/{self.experiment_name}/{g.name}_boundary_tsne.png'
            title = f'{g.name} model on {g.dataset.name}. Round {epoch + 1}.'
            plot_tsne_boundary(g.dataset._df, g.dataset._target, g.model, title, file_name, self.bounds)

            benchmark = self.benchmarks[g.name]
            results = self.experiment_data['evaluation'][g.name] = {}

            if benchmark._counterfactuals is None:
                results['success_rate'] = 0.0
                continue

            found_counterfactuals = benchmark._counterfactuals.index
            success_rate = len(found_counterfactuals) / len(benchmark._factuals)
            average_time = benchmark._timer / len(found_counterfactuals)

            results['success_rate'] = success_rate
            results['avg_time_per_ce'] = average_time

            benchmark._factuals = benchmark._factuals[benchmark._factuals.index.isin(found_counterfactuals)]
            log.info(f"Factuals: {benchmark._factuals.shape}, counterfactuals: {benchmark._counterfactuals.shape}")

            if benchmark._factuals.shape[0] == benchmark._counterfactuals.shape[0]:
                results['avg_classification_probability'] = float(benchmark._positive_class_proba.mean())
                results['avg_redundancy'] = benchmark.compute_redundancy().mean(axis=0).iloc[0]
                results['avg_ynn_of_counterfactual'] = benchmark.compute_ynn().mean(axis=0).iloc[0]
                results['avg_constraint_violation'] = benchmark.compute_constraint_violation().mean(axis=0).iloc[0]

                distances = benchmark.compute_distances().mean(axis=0)
                results['avg_changes_applied'] = distances.iloc[0]
                results['avg_taxicab_distance'] = distances.iloc[1]
                results['avg_euclidean_distance'] = np.sqrt(distances.iloc[2])
                results['avg_max_change_size'] = distances.iloc[3]

            # file_name = f'../experiment_data/{self.experiment_name}/{g.name}_dataset.csv'
            # g.dataset._df.to_csv(file_name, index=False, float_format='%1.4f')

    def save_data(self, path=None):
        """
        Write the data collected throughout the experiment into a .json file.

        Args:
            path (str):
                Directory where the dictionary of experiment data should be written.
        """
        path = path or f'../experiment_data/{self.experiment_name}/measurements.json'
        with open(path, 'w') as outfile:
            json.dump(self.experiment_data, outfile, sort_keys=True, indent=3)

    def describe(self):
        """
        Returns as dictionary the information about all generators under test.

        Returns:
            dict: Dictionary containing names and hyper-parameters of all generators.
        """
        result = {}
        for g in self.generators:
            result[g.name] = g.describe()
        return result

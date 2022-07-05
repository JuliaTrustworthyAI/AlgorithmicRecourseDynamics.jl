from .model import disagreement_distance, decisiveness, sample_MMD, model_MMD
from .distribution import measure_distribution, distribution_MMD
from .performance import measure_performance


def measure(generator, initial_data, calculate_p):
    """
    Quantify the dataset and model and save into `experiment_data`.

    Args:
        generator (RecourseGenerator):
            Recourse generator along with utilities required to conduct experiments.
        calculate_p (Boolean):
            If True, the statistical significance is calculated for MMD of distribution and model.

    Returns:
        dict: A dictionary storing all measurements for the current epoch.
    """
    results = {}

    # Measure the distributions of data
    results['distribution'] = measure_distribution(generator.dataset)

    # Measure the current performance of models
    results['performance'] = measure_performance(generator.dataset, generator.model)

    # Measure the disagreement between current model and the initial model
    results['disagreement'] = disagreement_distance(generator.dataset, generator.dataset.target,
                                                    initial_data['model'], generator.model)

    # Measure the average distance of a sample from the decision boundary
    results['decisiveness'] = decisiveness(generator.dataset, generator.model)

    # Measure the MMD of the distribution and the model
    results['MMD'] = distribution_MMD(generator.dataset, initial_data['samples'], calculate_p)
    results['sample_MMD'] = sample_MMD(generator.dataset, generator.model, initial_data['proba'], calculate_p)

    if generator.dataset._df.to_numpy().shape[1] <= 3:
        results['model_MMD'] = model_MMD(initial_data['grid'], initial_data['model'], generator.model, calculate_p)

    return results

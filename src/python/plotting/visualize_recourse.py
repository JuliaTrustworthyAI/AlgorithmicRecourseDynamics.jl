import imageio
import os


def generate_gif(experiment_path, generator_name):
    """
    Collect the images of data distribution over time and process them into a .gif file.

    Args:
        experiment_path (str): path to the experiment directory where images are stored
        generator_name (str): name of the generator which should be processed
    """
    # Find corresponding filenames
    filenames = [name for name in os.listdir(experiment_path) if os.path.isfile(os.path.join(experiment_path, name))]
    filenames = list(filter(lambda name: name.startswith(generator_name), filenames))

    # Generate the GIF and save to the original directory
    images = []
    for filename in sorted(filenames):
        for _ in range(3):
            images.append(imageio.imread(f'{experiment_path}/{filename}'))
    imageio.mimsave(f'{experiment_path}/{generator_name}.gif', images)

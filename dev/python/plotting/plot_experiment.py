import json
import matplotlib.pyplot as plt
import operator


from functools import reduce


def get_by_path(root, items):
    """
    Access a dictionary based on a set of keys in the provided order.

    Args:
        root (dict):
            Top-level of the nested dictionary.
        items (List[str]):
            List of strings specifying the consecutive keys.

    Returns:
        object: Value corresponding to the last key in the `items` list.

    """
    try:
        return reduce(operator.getitem, items, root)
    except Exception:
        raise ValueError("Specified path does not exist in the dictionary.")


def plot_experiment(output_directory, generators, plot_type, dict_path,
                    file_name='measurements.json', show_plot=False):
    """
    Plots a specified component of the experiment data gathered over all epochs of recourse.

    Args:
        output_directory (str):
            Name of the directory where images are saved.
        generators (List[str]):
            List of the names of all generators which should be plotted.
        plot_type (str):
            Type of the created plot.
        dict_path (List[str]):
            Location of the measurements of interest within the dictionary of experiment data.
        file_name (str):
            Name of the file containing the experiment data dictionary.
        show_plot (Boolean):
            If True the plot will also be outputted directly to the notebook.

    """
    with open(f'{output_directory}/{file_name}') as data_file:
        data = json.load(data_file)

        plt.figure(dpi=300)
        plt.grid(True)

        # Apply consistent theme over all plots generated for the project
        colormap = plt.cm.plasma
        colors = [colormap(int(g * colormap.N / len(generators))) for g in range(len(generators))]

        for index, g in enumerate(generators):
            # Check if the generators have been correctly specified
            if g not in data:
                raise ValueError(f'No measurements available for {g}')

            # Sort the keys in a dictionary of the generator in a chronological order
            data[g] = {int(k): v for k, v in data[g].items()}
            epochs = sorted(data[g].items())

            result = []
            for e in epochs:
                result.append(get_by_path(e[1], dict_path))

            plt.plot(range(len(result)), result, linewidth=2,
                     label=f'{g.capitalize()}', color=colors[index])

        # Format the plot
        plt.xlim([0, len(result) - 1])
        plt.ylim([0 - 0.2 * max(result), 1.2 * max(result)])
        plt.legend()
        plt.savefig(f"{output_directory}/{plot_type}.png", bbox_inches='tight')

        # Only show if asked
        if show_plot:
            plt.show()

        plt.close()

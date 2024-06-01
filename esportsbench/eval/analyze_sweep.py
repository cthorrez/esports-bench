import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    experiment_id = "experiment_1"
    results = defaultdict(dict)
    for game in os.listdir(f'../experiments/sweep_results/{experiment_id}'):
        for rs_file in os.listdir(f'../experiments/sweep_results/{experiment_id}/{game}'):
            rating_system = rs_file.removesuffix('.json')
            results[game][rating_system] = json.load(open(f'../experiments/sweep_results/{experiment_id}/{game}/{rs_file}'))

    # games = list(results.keys())
    # rating_systems = list(results[games[0]].keys())

    # Transform the data to be model-centric with dataset names
    transformed_data = {}
    for dataset, models in results.items():
        for model, data in models.items():
            params = data['best_params']
            if model not in transformed_data:
                transformed_data[model] = {}
            for param, value in params.items():
                if param not in transformed_data[model]:
                    transformed_data[model][param] = []
                transformed_data[model][param].append((dataset, value))

    # Plotting
    for model, params in transformed_data.items():
        for param, dataset_values in params.items():
            fig, ax = plt.subplots()
            # Sorting datasets alphabetically or by their original order if there's a specific one
            dataset_values.sort(key=lambda x: x[0])  # Sort by dataset name if necessary
            dataset_names = [dv[0] for dv in dataset_values]
            values = [dv[1] for dv in dataset_values]

            ax.plot(dataset_names, values, label=param, marker='o')
            ax.set_title(f'{model} - {param} across datasets')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Value')
            ax.set_xticklabels(dataset_names, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()

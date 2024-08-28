import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from esportsbench.arg_parsers import get_games_argparser, comma_separated
from esportsbench.eval.bench import run_benchmark, ALL_RATING_SYSTEMS
from cycler import cycler


def plot_metrics_vs_duration(data, durations):
    # Extract unique methods
    methods = set()
    for experiment in data:
        for dataset in experiment.values():
            methods.update(dataset.keys())
    methods = list(methods)

    # Calculate mean accuracy and log loss for each method and duration
    mean_metrics = {method: {'accuracy': [], 'log_loss': []} for method in methods}
    for experiment, duration in zip(data, durations):
        for method in methods:
            metrics = [dataset[method] for dataset in experiment.values() if method in dataset]
            if metrics:
                mean_metrics[method]['accuracy'].append(np.mean([m['accuracy'] for m in metrics]))
                mean_metrics[method]['log_loss'].append(np.mean([m['log_loss'] for m in metrics]))
            else:
                mean_metrics[method]['accuracy'].append(np.nan)
                mean_metrics[method]['log_loss'].append(np.nan)

    # Set up colorblind-friendly color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Accuracy plot
    for i, method in enumerate(methods):
        if method.endswith('_base'): continue
        ax1.plot(durations, mean_metrics[method]['accuracy'], marker=markers[i % len(markers)], label=method)
    ax1.set_xlabel('Rating Period Length (days)')
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_title('Mean Accuracy vs. Rating Period Length')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Log Loss plot
    for i, method in enumerate(methods):
        if method.endswith('_base'): continue
        ax2.plot(durations, mean_metrics[method]['log_loss'], marker=markers[i % len(markers)], label=method)
    ax2.set_xlabel('Rating Period Length (days)')
    ax2.set_ylabel('Mean Log Loss')
    ax2.set_title('Mean Log Loss vs. Rating Period Length')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('rating_period_plots.png', dpi=1200)
    plt.show()

def main(
    games,
    rating_systems,
    train_end_date,
    test_end_date,
    data_dir,
    drop_draws=None
):
    rating_periods = [1, 7, 14, 28]
    sweep_results_base = 'conf/sweep_results/fine_sweep'
    metrics_file = 'metrics.pkl'
    if os.path.exists(metrics_file):
        print(f"using cached metrics from {metrics_file}")
        metrics = pickle.load(open(metrics_file, 'rb'))
    else:
        print("computing metrics")
        metrics = []
        for rating_period in rating_periods:
            config_path = f"{sweep_results_base}_{rating_period}D_1000"
            print(f'getting test set results for {config_path}')
            results = run_benchmark(
                games=games,
                rating_systems=rating_systems,
                rating_period=f"{rating_period}D",
                train_end_date=train_end_date,
                test_end_date=test_end_date,
                data_dir=data_dir,
                drop_draws=drop_draws,
                hyperparameter_config=config_path
            )
            metrics.append(results)
            print(results)
        pickle.dump(metrics, open(metrics_file, 'wb'))

    plot_metrics_vs_duration(metrics, rating_periods)



if __name__ == '__main__':
    parser = get_games_argparser()
    parser.add_argument(
        '-rs',
        '--rating_systems',
        type=comma_separated(ALL_RATING_SYSTEMS),
        default=ALL_RATING_SYSTEMS,
    )
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-d', '--data_dir', type=str, default='hf_data')
    args = parser.parse_args()
    main(**vars(args))
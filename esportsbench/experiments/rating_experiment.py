"""
This script will run a full rating system experiment
  1. run a broad hyperparameter sweep
  2. run a fine hyperparameter sweep centered around the results of the broad sweep
  3. run the evaluation pipeline with the best hyperparameters identified by the fine sweep on the test set, report train and test numbers
"""
import warnings
import hydra
import pathlib
import yaml
from omegaconf import DictConfig
from esportsbench.eval.bench import run_benchmark, ALL_RATING_SYSTEMS, add_mean_metrics, print_results
from esportsbench.eval.sweep import sweep
from esportsbench.experiments.fine_sweep import construct_fine_sweep_config
from esportsbench.constants import GAME_SHORT_NAMES

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):

    param_file_path = pathlib.Path(__file__).parent / 'conf' / 'param_bounds.yaml'
    param_bounds = yaml.full_load(open(param_file_path))
    
    games = GAME_SHORT_NAMES if config.games == 'all' else config.games
    rating_systems = ALL_RATING_SYSTEMS if config.rating_systems == 'all' else config.rating_systems

    common_sweep_args = {
        'data_dir' : config.data_dir,
        'rating_period': config.rating_period,
        'train_end_date' : config.train_end_date,
        'test_end_date' : config.test_end_date,
        'num_samples' : config.num_samples,
        'num_processes' : config.num_processes,
    }

    best_params = {}
    for game in games:
        broad_sweep_results = sweep(
            games=[game],
            rating_systems=config.rating_systems,
            granularity='broad',
            sweep_config=config.broad_sweep_config,
            **common_sweep_args
        )

        fine_sweep_config = construct_fine_sweep_config(broad_sweep_results, param_bounds)
        fine_sweep_results = sweep(
            games=[game],
            rating_systems=config.rating_systems,
            granularity='fine',
            sweep_config=fine_sweep_config,
            **common_sweep_args,
        )
        best_params[game] = fine_sweep_results[game]

    benchark = run_benchmark(
        games=games,
        rating_systems=rating_systems,
        rating_period=config.rating_period,
        train_end_date=config.train_end_date,
        test_end_date=config.test_end_date,
        data_dir=config.data_dir,
        hyperparameter_config=best_params
    )
    print_results(add_mean_metrics(benchark))
        

if __name__ == '__main__':
    main()
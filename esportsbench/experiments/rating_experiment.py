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
from esportsbench.eval.bench import run_benchmark, add_mean_metrics, print_results
from esportsbench.eval.sweep import sweep
from esportsbench.eval.experiment_config import HyperparameterConfig
from esportsbench.constants import GAME_SHORT_NAMES, ALL_RATING_SYSTEM_NAMES

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)


def construct_fine_sweep_config(broad_sweep_config, param_bounds, low_multiplier=0.75, high_multiplier=1.25):
    fine_sweep_config = {}
    for game, game_config in broad_sweep_config.items():
        fine_game_config = {}
        for rating_system, best_params in game_config.items():
            fine_rating_system_config = {}
            for param_name, best_value in best_params.items():
                if (param_name == "model") or (isinstance(best_value, str)): continue
                lower_bound = upper_bound = None
                if (rating_system in param_bounds) and (param_name in param_bounds[rating_system]):
                    if param_bounds[rating_system][param_name].get('param_type') == 'list':
                        fine_rating_system_config[param_name] = HyperparameterConfig(param_type='list', options=[best_value])
                        continue
                    lower_bound = param_bounds[rating_system][param_name].get('lower_bound')
                    upper_bound = param_bounds[rating_system][param_name].get('upper_bound')
                sweep_min_value = best_value * low_multiplier
                sweep_max_value = best_value * high_multiplier
                if lower_bound is not None:
                    sweep_min_value = max(lower_bound, sweep_min_value)
                if upper_bound is not None:
                    sweep_max_value = min(upper_bound, sweep_max_value)
                fine_rating_system_config[param_name] = HyperparameterConfig(min_value=sweep_min_value, max_value=sweep_max_value)
            fine_game_config[rating_system] = fine_rating_system_config
        fine_sweep_config[game] = fine_game_config
    return fine_sweep_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):

    param_file_path = pathlib.Path(__file__).parent / 'conf' / 'param_bounds.yaml'
    param_bounds = yaml.full_load(open(param_file_path))
    
    games = GAME_SHORT_NAMES if config.games == 'all' else config.games
    
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
            granularity='broad',
            sweep_config=config.broad_sweep_config,
            **common_sweep_args
        )

        fine_sweep_config = construct_fine_sweep_config(broad_sweep_results, param_bounds)
        for rating_key in config.broad_sweep_config:
            for game in fine_sweep_config:
                fine_sweep_config[game][rating_key]['model'] = config.broad_sweep_config[rating_key]['model']

        fine_sweep_results = sweep(
            games=[game],
            granularity='fine',
            sweep_config=fine_sweep_config,
            **common_sweep_args,
        )
        best_params[game] = fine_sweep_results[game]


    for rating_key in config.broad_sweep_config:
        for game in best_params:
            best_params[game][rating_key]['model'] = config.broad_sweep_config[rating_key]['model']

    benchark = run_benchmark(
        games=games,
        rating_period=config.rating_period,
        train_end_date=config.train_end_date,
        test_end_date=config.test_end_date,
        data_dir=config.data_dir,
        hyperparameter_config=best_params
    )
    print_results(benchark)        

if __name__ == '__main__':
    main()
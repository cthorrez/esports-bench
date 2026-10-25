import os
import yaml
import json
import pathlib
import warnings
from omegaconf import DictConfig
from esportsbench.eval.experiment_config import load_config
from esportsbench.eval.sweep import sweep
from esportsbench.constants import GAME_SHORT_NAMES
from esportsbench.eval.experiment_config import HyperparameterConfig

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_sweep_results(sweep_results_dir):
    sweep_config = {}
    games = os.listdir(sweep_results_dir)
    for game in games:
        game_config = {}
        file_names = os.listdir(f'{sweep_results_dir}/{game}')
        for file_name in file_names:
            rating_system = file_name.removesuffix('.json')
            params = json.load(open(f'{sweep_results_dir}/{game}/{file_name}', 'r'))
            game_config[rating_system] = params['best_params']
        sweep_config[game] = game_config
    return sweep_config

def construct_fine_sweep_config(broad_sweep_config, param_bounds, low_multiplier=0.75, high_multiplier=1.25):
    fine_sweep_config = {}
    for game, game_config in broad_sweep_config.items():
        fine_game_config = {}
        for rating_system, best_params in game_config.items():
            fine_rating_system_config = {}
            for param_name, best_value in best_params.items():
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

def main(config: DictConfig):
    if config.games == 'all':
        games = GAME_SHORT_NAMES
    else:
        games = config.games
    conf_dir = pathlib.Path(__file__).parent / 'conf'
    param_bounds = yaml.full_load(open(conf_dir / 'param_bounds.yaml'))
    # relative paths are resolved against conf/sweep_results, e.g. broad_sweep_7D_1000
    sweep_results_path = conf_dir / 'sweep_results' / config.sweep_results_path
    broad_sweep_results = load_sweep_results(sweep_results_path)
    sweep_config = construct_fine_sweep_config(broad_sweep_results, param_bounds)
    # the broad sweep results only store params, copy over which model each rating system key uses
    for game_config in sweep_config.values():
        for rating_system_key, rating_system_config in game_config.items():
            if rating_system_key in config.broad_sweep_config:
                rating_system_config['model'] = config.broad_sweep_config[rating_system_key]['model']
            else:
                rating_system_config['model'] = rating_system_key

    sweep(
        games=games,
        rating_systems=config.get('rating_systems', 'all'),
        data_dir=config.data_dir,
        granularity='fine',
        sweep_config=sweep_config,
        rating_period=config.rating_period,
        train_end_date=config.train_end_date,
        test_end_date=config.test_end_date,
        num_samples=config.num_samples,
        num_processes=config.num_processes,
    )


if __name__ == '__main__':
    main(load_config())
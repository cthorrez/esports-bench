import os
import yaml
import json
import pathlib
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
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
                if param_name in param_bounds[rating_system]:
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
    if config.games == 'all':
        games = GAME_SHORT_NAMES
    else:
        games = config.games
    param_file_path = pathlib.Path(__file__).parent / 'conf' / 'param_bounds.yaml'
    param_bounds = yaml.full_load(open(param_file_path))
    broad_sweep_results = load_sweep_results(config.sweep_results_path)
    sweep_config = construct_fine_sweep_config(broad_sweep_results, param_bounds)

    for game in games:
        sweep(
            games=[game],
            rating_systems=config.rating_systems,
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
    main()
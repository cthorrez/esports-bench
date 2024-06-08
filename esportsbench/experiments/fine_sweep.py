import os
import yaml
import json
import pathlib
import warnings
from dacite import from_dict
from esportsbench.eval.sweep_config import ExperimentSweepConfig
from esportsbench.arg_parsers import get_games_argparser
from esportsbench.eval.sweep import sweep

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)



def construct_config(config_dir, param_bounds):
    """
    given the results of a broad sweep, construct the narrow sweep configs
    """
    config = {}
    for root, dirs, files in os.walk(config_dir):
        path_parts = root.split(os.sep)
        if len(path_parts) > 1:
            game = path_parts[-1]
            config[game] = {}
            for file in files:
                if file.endswith('.json'):
                    rating_system = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        best_params = json.load(f)['best_params']
                        sweep_config = {}
                        for param_name, best_value in best_params.items():
                            if param_name in param_bounds[rating_system]:
                                if param_bounds[rating_system][param_name].get('param_type') == 'list':
                                    sweep_config[param_name] = {'param_type' : 'list', 'values' : [best_value]}
                                    continue
                                lower_bound = param_bounds[rating_system][param_name].get('lower_bound')
                                upper_bound = param_bounds[rating_system][param_name].get('upper_bound')
                            sweep_min_value = best_value * 0.75
                            sweep_max_value = best_value * 1.25
                            if lower_bound is not None:
                                sweep_min_value = max(lower_bound, sweep_min_value)
                            if upper_bound is not None:
                                sweep_max_value = min(upper_bound, sweep_max_value)
                            sweep_config[param_name] = {'min_value': sweep_min_value, "max_value": sweep_max_value}
                        config[game][rating_system] = sweep_config
    return config




def main(
    games,
    data_dir,
    config_dir,
    drop_draws=False,
    rating_period='7D',
    train_end_date='2023-03-31',
    test_end_date='2024-03-31',
    num_samples=100,
    num_processes=8,
):
    param_file_path = pathlib.Path(__file__).parent / 'configs' / 'param_bounds.yaml'
    param_bounds = yaml.full_load(open(param_file_path))
    config_dict = construct_config(config_dir, param_bounds)

    for game in games:
        config = from_dict(data_class=ExperimentSweepConfig, data={'param_configs' : config_dict[game]})

        sweep(
            [game],
            config,
            data_dir=data_dir,
            granularity='fine',
            drop_draws=drop_draws,
            rating_period=rating_period,
            train_end_date=train_end_date,
            test_end_date=test_end_date,
            num_samples=num_samples,
            num_processes=num_processes,
        )


if __name__ == '__main__':
    parser = get_games_argparser()
    parser.add_argument('-d', '--data_dir', type=str, default='hf_data')
    parser.add_argument('-c', '--config_dir', type=str, default='sweep_results/broad_sweep_7D_1000', required=False)
    parser.add_argument('-rp', '--rating_period', type=str, required=False, default='7D')
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-ns', '--num_samples', type=int, default=1000, required=False)
    parser.add_argument('-np', '--num_processes', type=int, default=12, required=False)
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
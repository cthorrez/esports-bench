"""module for sweeping hyperparameters"""
import os
import pathlib
import json
import warnings
from typing import Dict
import numpy as np
from riix.eval import grid_search
from esportsbench.datasets import load_dataset
from esportsbench.eval.bench import RATING_SYSTEM_MAP
from esportsbench.eval.experiment_config import HyperparameterConfig

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)


def construct_param_configurations(
    param_configs,
    num_samples: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    param_values = {}
    for param_name, param_config in param_configs.items():
        
        if (hasattr(param_config, 'param_type')) and (getattr(param_config, 'param_type') == 'list'):
            idxs = rng.choice(len(param_config.options), size=num_samples)
            # doing list instead of array since this could be various dtypes
            sampled_values = [param_config.options[int(idx)] for idx in idxs]
        else:
            sampled_values = rng.uniform(low=param_config.min_value, high=param_config.max_value, size=num_samples)

        param_values[param_name] = sampled_values

    param_samples = []
    for sample_idx in range(num_samples):
        sample = {}
        for param_name in param_configs.keys():
            sample[param_name] = param_values[param_name][sample_idx]
        param_samples.append(sample)
    return param_samples


def sweep(
    games,
    data_dir,
    granularity,
    sweep_config,
    train_end_date,
    test_end_date,
    rating_systems='all',
    rating_period='7D',
    drop_draws=False,
    num_samples=100,
    num_processes=8,
):
    results_dir = pathlib.Path(__file__).parents[1] / 'experiments' / 'sweep_results' / f'{granularity}_sweep_{rating_period}_{num_samples}'
    os.makedirs(results_dir, exist_ok=True)
    for dataset_name in games:
        dataset, test_mask = load_dataset(
            dataset_name,
            rating_period=rating_period,
            drop_draws=drop_draws,
            train_end_date=train_end_date,
            test_end_date=test_end_date,
            data_dir=data_dir,
        )


        train_rows = int(np.logical_not(test_mask).sum())
        dataset = dataset[:train_rows]
        train_mask = np.ones(len(dataset), dtype=np.bool_)
        print(f'Sweeping on dataset with {len(dataset)} rows')

        if rating_systems != 'all':
            sweep_config = {key : val for key,val in sweep_config.items() if key in rating_systems}

        for rating_system_name in sweep_config.keys():
            print(f'Sweeping {rating_system_name} on {dataset_name} over {num_samples} configurations')
            rating_system_class = RATING_SYSTEM_MAP[rating_system_name]
            param_configurations = construct_param_configurations(
                param_configs=sweep_config[rating_system_name], num_samples=num_samples
            )
            best_params, best_metrics = grid_search(
                rating_system_class=rating_system_class,
                dataset=dataset,
                metrics_mask=train_mask, # during sweeping the train data is the val data
                param_configurations=param_configurations,
                metric='log_loss',
                minimize_metric=True,
                num_processes=num_processes,
            )
            print(best_params)
            print(best_metrics)
            out_dict = {'best_params': best_params.copy(), 'best_metrics': best_metrics.copy()}
            out_dir = f'{results_dir}/{dataset_name}'
            os.makedirs(out_dir, exist_ok=True)
            out_file_path = f'{out_dir}/{rating_system_name}.json'
            json.dump(out_dict, open(out_file_path, 'w'), indent=2)
            del best_metrics, best_params


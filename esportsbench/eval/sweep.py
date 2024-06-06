"""module for sweeping hyperparameters"""
import os
import math
import yaml
import json
import warnings
from typing import Dict
import numpy as np
import pandas as pd
from dacite import from_dict
from riix.eval import grid_search
from esportsbench.datasets import load_dataset
from esportsbench.eval.bench import RATING_SYSTEM_MAP
from esportsbench.eval.sweep_config import ExperimentSweepConfig, ParamSweepConfig
from esportsbench.arg_parsers import get_games_argparser

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)


def construct_param_configurations(
    param_configs: Dict[str, ParamSweepConfig],
    num_samples: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    param_values = {}
    for param_name, param_config in param_configs.items():
        if param_config.param_type == 'range':
            sampled_values = rng.uniform(low=param_config.min_value, high=param_config.max_value, size=num_samples)
        elif param_config.param_type == 'list':
            idxs = rng.choice(len(param_config.values), size=num_samples)
            # doing list instead of array since this could be various dtypes
            sampled_values = [param_config.values[idx] for idx in idxs]
        else:
            raise ValueError(f'param_type should be "range" or "list", got {param_config.param_type}')

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
    config,
    granularity,
    drop_draws=False,
    rating_period='7D',
    train_end_date='2023-03-31',
    test_end_date='2024-03-31',
    num_samples=100,
    num_processes=8,
):
    results_dir = f'sweep_results/{granularity}_sweep_{rating_period}_{num_samples}'
    os.makedirs(results_dir, exist_ok=True)
    for dataset_name in games:
        dataset, test_mask = load_dataset(
            dataset_name,
            rating_period=rating_period,
            drop_draws=drop_draws,
            train_end_date=train_end_date,
            test_end_date=test_end_date
        )


        train_rows = int(np.logical_not(test_mask).sum())
        dataset = dataset[:train_rows]
        train_mask = np.ones(len(dataset), dtype=np.bool_)
        print(f'Sweeping on dataset with {len(dataset)} rows')

        for rating_system_name in config.param_configs.keys():
            print(f'Sweeping {rating_system_name} on {dataset_name} over {num_samples} configurations')
            rating_system_class = RATING_SYSTEM_MAP[rating_system_name]
            param_configurations = construct_param_configurations(
                param_configs=config.param_configs[rating_system_name], num_samples=num_samples
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


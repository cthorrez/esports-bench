import pathlib
from dataclasses import dataclass
from typing import Literal, List, Optional, Union
from omegaconf import OmegaConf, DictConfig

CONF_DIR = pathlib.Path(__file__).resolve().parents[1] / 'experiments' / 'conf'


def load_config(args: list[str] | None = None) -> DictConfig:
    """load an experiment config from experiments/conf with key=value command line overrides
    (replaces hydra, whose argparse setup crashes on python 3.14)

    examples:
        python broad_sweep.py games=[tetris] num_samples=4
        python broad_sweep.py config_name=trueskill_time_dynamics
        python broad_sweep.py broad_sweep_config=trueskill_time_dynamics
    """
    overrides = OmegaConf.from_cli(args)
    config_name = overrides.pop('config_name', 'config')
    config = OmegaConf.load(CONF_DIR / f'{config_name}.yaml')
    # resolve the hydra style defaults list, e.g. `- broad_sweep_config: default` loads conf/broad_sweep_config/default.yaml
    for default in config.pop('defaults', []):
        if default == '_self_':
            continue
        for group, name in default.items():
            name = overrides.pop(group, name)
            config[group] = OmegaConf.load(CONF_DIR / group / f'{name}.yaml')
    return OmegaConf.merge(config, overrides)

@dataclass
class HyperparameterConfig:
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List] = None
    param_type: Literal['range', 'list'] = 'range'


@dataclass
class ExperimentConfig:
    # key levels:
    # * game (only present if fine sweep)
    # * rating_system
    # * param_name
    broad_sweep_config: dict = None

    # set this if doing a fine sweep
    sweep_results_path: str = None

    data_dir: str = 'data/final_data'
    train_end_date: str = '2023-03-31'
    test_end_date: str = '2024-03-31'
    rating_period: str = '7D'
    num_samples: int = 1000
    num_processes: int = 8
    games: Union[Literal['all'], List[str]] = 'all'
    rating_systems: Union[Literal['all'], List[str]] = 'all'


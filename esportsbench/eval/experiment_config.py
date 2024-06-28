from dataclasses import dataclass
from typing import Literal, List, Optional, Dict, Union

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
    sweep_config: dict = None

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


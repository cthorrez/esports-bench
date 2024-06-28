from dataclasses import dataclass
from typing import Literal, List, Optional, Dict, Union

@dataclass
class HyperparameterConfig:
    min_value: Optional[float]
    max_value: Optional[float]
    options: Optional[List]
    param_type: Literal['range', 'list'] = 'range'


@dataclass
class ExperimentConfig:
    # first level key is rating_system
    # second level key is param_name
    sweep_config: Dict[str, Dict[str, HyperparameterConfig]]

    data_dir: str = 'data/final_data'
    train_end_date: str = '2023-03-31'
    test_end_date: str = '2024-03-31'
    rating_period: str = '7D'
    num_samples: int = 1000
    num_processes: int = 8
    games: Union[Literal['all'], List[str]] = 'all'
    rating_systems: Union[Literal['all'], List[str]] = 'all'


from dataclasses import dataclass
from typing import Literal, List, Optional, Dict, Union


@dataclass
class ExperimentConfig:
    train_end_date: str
    test_end_date: str
    num_samples: int
    num_processes: int = 8
    games: Union[Literal['all'], List[str]]
    rating_systems: Union[Literal['all'], List[str]]

@dataclass
class ParamSweepConfig:
    min_value: Optional[float]
    max_value: Optional[float]
    values: Optional[List]
    param_type: Literal['range', 'list'] = 'range'


@dataclass
class ExperimentSweepConfig:
    rating_period: Optional[str]
    num_samples: Optional[int]
    # first level key is rating_system
    # second level key is param_name
    param_configs: Dict[str, Dict[str, ParamSweepConfig]]

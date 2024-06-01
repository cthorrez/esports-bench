from dataclasses import dataclass
from typing import Literal, List, Optional, Dict


@dataclass
class ParamSweepConfig:
    min_value: Optional[float]
    max_value: Optional[float]
    values: Optional[List]
    param_type: Literal['range', 'list'] = 'range'


@dataclass
class ExperimentSweepConfig:
    experiment_id: str
    # first level key is rating_system
    # second level key is param_name
    param_configs: Dict[str, Dict[str, ParamSweepConfig]]

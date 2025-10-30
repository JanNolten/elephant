from typing import (
    Any,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

import elephant.schemas.field_validator as fv

class PydanticTotalSpikingProbability(BaseModel):
    """
    PyDantic Class to wrap the elephant.functional_connectivity.total_spiking_probability function
    with additional type checking and json_schema by PyDantic.
    """
    
    spike_trains: Any = Field(..., description="Binned spike train")
    surrounding_window_sizes: list[int] = Field([3, 4, 5, 6, 7, 8], description="Array of window sizes for surrounding area")
    observed_window_sizes: list[int] = Field([2, 3, 4, 5, 6], description="Array of window sizes for observed area")
    crossover_window_sizes: list[int] = Field([0], description="Array of window sizes for crossover")
    max_delay: int = Field(25, gt=0, description="max delay when performing normalized cross-correlations")
    normalize: bool = Field(False, description="Normalize output")

    
    @field_validator("spike_trains")
    @classmethod
    def validate_neo_BinnedSpikeTrain(cls, v, info):
        return fv.validate_binned_spiketrain(v, info)
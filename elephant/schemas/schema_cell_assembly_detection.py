import quantities as pq
import numpy as np
from typing import (
    Any,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

class PydanticCellAssemblyDetection(BaseModel):
    """
    PyDantic Class to wrap the elephant.cell_assembly_detection.cell_assembly_detection function
    with additional type checking and json_schema by PyDantic.
    """

    binned_spiketrain: Any = Field(..., description="Binned spike train")
    max_lag: int = Field(..., ge=0, description="Maximal lag")
    reference_lag: Optional[int] = Field(2, ge=0, description="Reference lag for non-stationarity correction")
    alpha: Optional[float] = Field(0.05, gt=0, le=1, description="Significance level for statistical test")
    min_occurrences: Optional[int] = Field(0, ge=0, description="Minimal number of occurrences required")
    size_chunks: Optional[int] = Field(100, gt=0, description="Chunk size for processing")
    max_spikes: Optional[int] = Field(np.inf, ge=1, description="Maximal assembly order")
    signifiance_pruning: Optional[bool] = Field(True, description="Perform signifiance pruning")
    subgroup_pruning: Optional[bool] = Field(True, description="Perform subgroup pruning")
    same_configuration_pruning: Optional[bool] = Field(False, description="Perform pruning")
    verbose: Optional[bool] = Field(False, description="Give all prints")

    @field_validator("binned_spiketrain")
    @classmethod
    def validate_binned_spiketrain(cls, v, info):
        return fv.validate_binned_spiketrain(v, info)
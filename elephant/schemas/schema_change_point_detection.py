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
import neo

import elephant.schemas.field_validator as fv

class PydanticMultipleFilterTest(BaseModel):
    """
    Pydantic wrapper for elephant.change_point_detection.multiple_filter_test
    """

    window_sizes: list = Field(..., description="List of window sizes (quantities)")
    spiketrain: Any = Field(..., description="Spike train (neo.SpikeTrain or quantity array)")
    t_final: Any = Field(..., ge=0, description="Final time")
    alpha: float = Field(..., ge=0.0, le=100.0, description="Alpha-quantile in [0,100]")
    n_surrogates: Optional[int] = Field(1000, ge=0, description="Number of simulated limit processes")
    test_quantile: Optional[float] = Field(None, ge=0,description="Optional threshold for maxima of filter derivative processes")
    test_param: Optional[Any] = Field(None, description="Optional test parameters (np.ndarray)")
    time_step: Optional[Any] = Field(None, ge=0, description="Time step (quantity) for sliding windows")

    # Validators
    @field_validator("window_sizes")
    @classmethod
    def validate_window_sizes(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(list,), allow_none=False, min_length=1, allowed_content_types=(pq.Quantity,))

    @field_validator("spiketrain")
    @classmethod
    def validate_spiketrain(cls, v, info):
        return fv.validate_spiketrain(v, info, allowed_types=(neo.SpikeTrain, pq.Quantity), allow_none=False)

    @field_validator("t_final")
    @classmethod
    def validate_quantity(cls, value, info):
        return fv.validate_quantity(value, info)
    
    @field_validator("test_param")
    @classmethod
    def validate_test_param(cls, value, info):
        return fv.validate_array(value, info, allowed_types=(np.ndarray,), allow_none=True, min_length=1)
    
    @field_validator("time_step")
    @classmethod
    def validate_time_step(cls, value, info):
        return fv.validate_quantity(value, info, allow_none=True)


class PydanticEmpiricalParameters(BaseModel):
    """
    Pydantic wrapper for elephant.change_point_detection.empirical_parameters
    """

    window_sizes: Any = Field(..., description="List of window sizes")
    t_final: Any = Field(..., ge=0, description="Final time ")
    alpha: float = Field(..., ge=0.0, le=100.0, description="Alpha-quantile in [0,100]")
    n_surrogates: Optional[int] = Field(1000, ge=0, description="Number of simulated limit processes")
    time_step: Optional[Any] = Field(None, ge=0, description="Time step for sliding windows")

    # Validators
    @field_validator("window_sizes")
    @classmethod
    def validate_window_sizes(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(list,), allow_none=False, min_length=1, allowed_content_types=(pq.Quantity,))

    @field_validator("t_final")
    @classmethod
    def validate_times(cls, value, info):
        return fv.validate_quantity(value, info)
    
    @field_validator("time_step")
    @classmethod
    def validate_time_step(cls, value, info):
        return fv.validate_quantity(value, info, allow_none=True)


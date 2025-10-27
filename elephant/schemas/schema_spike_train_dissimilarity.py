import quantities as pq
from typing import (
    Any,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    field_serializer
)
import neo
from enum import Enum
import elephant

import field_validator as fv
import field_serializer as fs


class PydanticVictorPurpuraDistance(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_dissimilarity.victor_purpura_distance function
    with additional type checking and json_schema by PyDantic.
    """

    class AlgorithmOptions(Enum):
        fast = "fast"
        intuitive = "intuitive"

    spiketrains: list = Field(..., description="List of Spiketrains")
    cost_factor: Optional[Any] = Field(default_factory=lambda: 1. * pq.Hz, description="Cost factor for spike shifts")
    kernel: Optional[Any] = Field(None, description="Kernel to use in distance calculation")
    sort: Optional[bool] = Field(True, description="Sort Spiketrains")
    algorithm: Optional[AlgorithmOptions] = Field(AlgorithmOptions.fast, description="Which algorithm to calculate")

    @field_serializer("cost_factor", mode='plain')
    def serialize_quantity(self, value: pq.Quantity) -> dict:
        return fs.serialize_quantity(value)

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, spiketrains, info):
        return fv.validate_spiketrains(spiketrains, info, allowed_content_types=(neo.core.SpikeTrain,))

    @field_validator("cost_factor")
    @classmethod
    def validate_cost_factor(cls, cost_factor, info):
        return fv.validate_quantity(cost_factor, info, allow_none=False)

    @field_validator("kernel")
    @classmethod
    def validate_kernel(cls, kernel, info):
        # Kernel can be None or an elephant.kernels.Kernel
        return fv.validate_type(kernel, info, (elephant.kernels.Kernel,), allow_none=True)


class PydanticVanRossumDistance(BaseModel):
    """
    PyDantic Class to wrap the elephant.spike_train_dissimilarity.van_rossum_distance function
    with additional type checking and json_schema by PyDantic.
    """

    spiketrains: list = Field(..., description="List of Spiketrains")
    time_constant: Any = Field(default_factory=lambda: 1. * pq.s, description="Decay rate of exponential function")
    sort: bool = Field(True, description="Sort Spiketrains")

    @field_serializer("time_constant", mode='plain')
    def serialize_quantity(self, value: pq.Quantity) -> dict:
        return fs.serialize_quantity(value)

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, spiketrains, info):
        return fv.validate_spiketrains(spiketrains, info, allowed_content_types=(neo.core.SpikeTrain,))

    @field_validator("time_constant")
    @classmethod
    def validate_time_constant(cls, time_constant, info):
        return fv.validate_quantity(time_constant, info, allow_none=False)




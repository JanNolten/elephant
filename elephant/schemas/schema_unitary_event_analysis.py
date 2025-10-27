import quantities as pq
from typing import (
    Any,
    Union,
    Self,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    field_serializer
)
import neo
from enum import Enum

import field_validator as fv
import field_serializer as fs


class PydanticJointJWindowAnalysis(BaseModel):
    """
    PyDantic Class to wrap the elephant.unitary_event_analysis.jointJ_window_analysis function
    with additional type checking and json_schema by PyDantic.
    """

    class MethodOptions(Enum):
        analytic_TrialByTrial = "analytic_TrialByTrial"
        analytic_TrialAverage = "analytic_TrialAverage"
        surrogate_TrialByTrial = "surrogate_TrialByTrial"

    spiketrains: list = Field(..., description="List of Spiketrains")
    bin_size: Optional[Any] = Field(default_factory=5 * pq.ms, ge=0, description="The size of bins")
    win_size: Optional[Any] = Field(default_factory=100 * pq.ms, ge=0, description="The size of window")
    win_step: Optional[Any] = Field(default_factory=5 * pq.ms, ge=0, description="The size of window step")
    pattern_hash: Optional[Union[int, list]] = Field(None, description="Interested patterns in hash values")
    method: Optional[MethodOptions] = Field(MethodOptions.analytic_TrialByTrial, description="Method to compute unitary events")
    t_start: Optional[Any] = Field(None, description="Start time")
    t_stop: Optional[Any] = Field(None, description="Stop time")
    binary: Optional[bool] = Field(True, description="Binarize the binned spike trains")
    n_surrogates: Optional[int] = Field(100, ge=0, description="Number of surrogates used")

    @field_serializer("bin_size", "win_size", "win_step", mode='plain')
    def serialize_quantity(self, value: pq.Quantity) -> dict:
        return fs.serialize_quantity(value)
    
    @field_validator("bin_size", "win_size", "win_step")
    @classmethod
    def validate_quantity(cls, quantity, ctx):
        return fv.validate_quantity(quantity, ctx)
    
    @field_validator("t_start", "t_stop")
    @classmethod
    def validate_time(cls, time, ctx):
        return fv.validate_time(time, ctx)

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, spiketrains, info):
        return fv.validate_spiketrains(spiketrains, info, allowed_content_types=(neo.core.SpikeTrain,))
    
    @field_validator("binary")
    @classmethod
    def validate_binary(cls, binary):
        if not binary:
            raise NotImplementedError("The method only works if binary is True")
        return binary
    
    @model_validator(mode="after")
    def validate_model(self) -> Self:             
        winsize_bintime = self.win_size.magnitude // self.bin_size.magnitude
        winstep_bintime = self.win_step.magnitude // self.bin_size.magnitude

        if winsize_bintime * self.bin_size.magnitude != self.win_size.magnitude:
            raise UserWarning(f"The ratio between the win_size and the bin_size is not an integer")

        if winstep_bintime * self.bin_size.magnitude != self.win_step.magnitude:
            raise UserWarning(f"The ratio between the win_step and the bin_size is not an integer")
        return self
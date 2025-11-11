import quantities as pq
from typing import (
    Any,
    Union,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    field_serializer,
    model_validator,
)
import neo
from enum import Enum
import numpy as np
import elephant
import scipy.sparse as sp

import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

import warnings

from elephant.spike_train_surrogates import SURR_METHODS

class PydanticSurrogates(BaseModel, extra='allow'):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.surrogates` function
    with additional type checking and JSON schema generation.
    """
    spiketrain: Any = Field(..., description="Spiketrain Object(s)")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of surrogates")
    method: Optional[str] = Field("dither_spike_train", description="method to use generate surrogate spike trains")
    dt: Optional[Any] = Field(None, description="size of shift/window")

    @field_validator("spiketrain")
    @classmethod
    def validate_spiketrain(cls, value, info):
        if(isinstance(value,list)):
            return fv.validate_spiketrains(value, info, allowed_content_types=(neo.SpikeTrain,))
        return fv.validate_spiketrain(value, info,)
    
    @field_validator("method")
    @classmethod
    def validate_surr_method(cls, value, info):
        return fv.validate_key_in_tuple(value, info, SURR_METHODS)
    
    @field_validator("dt")
    @classmethod
    def validate_quantity(cls, value, info):
        return fv.validate_quantity(value, info, allow_none=True)
    
    @model_validator(mode="after")
    def validate_model(self):             
        if isinstance(self.spiketrain, list) and self.method != "trial_shifting":
            raise ValueError("spiketrain is only allowed to be a list when the method is trial_shifting")
        if self.dt is None and self.method is not "randomise_spikes" and self.method is not "shuffle_isis":
            raise ValueError("dt cannot be None if the method is not \"randomise_spikes\" nor \"shuffle_isis\"")
        return self
    

class PydanticJointISI(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.JointISI` function
    with additional type checking and JSON schema generation.
    """

    class MethodOptions(Enum):
        fast="fast"
        window="window"

    spiketrain: Any = Field(..., description="Spiketrain Object")
    dither: Optional[Any] = Field(default_factory=lambda: 15. * pq.ms, description="maximum displacement of a spike")
    truncation_limit: Optional[Any] = Field(default_factory=lambda: 100. * pq.ms, description="truncation limit")
    n_bins: Optional[int] = Field(100, gt=0, description="Size of joint-ISI-distribution")
    sigma: Optional[Any] = Field(default_factory=lambda: 2. * pq.ms, description="standard deviation of Gaussian kernel")
    alternate: Optional[bool] = Field(True, description="Dither even spikes than odd spikes")
    use_sqrt: Optional[bool] = Field(False, description="Preprocess joint-ISI histogram by square root")
    method: Optional[MethodOptions] = Field(MethodOptions.window, description="method option")
    cutoff: Optional[bool] = Field(True, description="Limit filetering of Joint-ISI histogram")
    refractory_period: Optional[Any] = Field(default_factory=lambda: 4. * pq.ms, description="refractory period")
    isi_dithering: Optional[bool] = Field(False, description="Destroy all serial correlations")


    @field_serializer("dither", "truncation_limit", "sigma", "refractory_period", mode='plain')
    def serialize_quantity(self, v):
        return fs.serialize_quantity(v)

    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    
    @field_validator("dither", "truncation_limit", "sigma", "refractory_period")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
    

class PydanticDitherSpikes(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.dither_spikes` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    dither: Any = Field(..., description="maximum displacement of a spike")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")
    decimals: Optional[Union[int, None]] = Field(None, ge=0, description="Number of decimal points")
    edges: Optional[bool] = Field(True, description="Drop out surrogate spikes outside range")
    refractory_period: Optional[Any] = Field(None, description="refractory period")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    
    @field_validator("dither")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
    
    @field_validator("refractory_period")
    @classmethod
    def validate_quantity_None(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)
    

class PydanticRandomiseSpikes(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.randomise_spikes` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")
    decimals: Optional[Union[int, None]] = Field(None, ge=0, description="Number of decimal points")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    

class PydanticShuffleIsis(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.shuffle_isis` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")
    decimals: Optional[Union[int, None]] = Field(None, ge=0, description="Number of decimal points")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    
class PydanticDitherSpikeTrain(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.dither_spike_train` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    shift: Any = Field(..., description="maximum displacement of a spike")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")
    decimals: Optional[Union[int, None]] = Field(None, ge=0, description="Number of decimal points")
    edges: Optional[bool] = Field(True, description="Drop out surrogate spikes outside range")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    
    @field_validator("shift")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
    

class PydanticJitterSpikes(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.jitter_spikes` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    bin_size: Any = Field(..., description="Size of time bins")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    
    @field_validator("bin_size")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
    

class PydanticBinShuffling(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.bin_shuffling` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    max_displacement: int = Field(..., ge=0, description="Number of bins spike can be displaced")
    bin_size: Any = Field(None, description="Size of time bins")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")
    sliding: Optional[bool] = Field(False, description="Slide window bin by bin")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain, elephant.conversion.BinnedSpikeTrain))
    
    @field_validator("bin_size")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)
    
    @model_validator(mode="after")
    def validate_model(self):             
        if self.sliding and not isinstance(self, elephant.conversion.BinnedSpikeTrain):
            warnings.warn("sliding is only implemented for binned spike trains", UserWarning)
        return self
    

class PydanticTrialShifting(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.trial_shifting` function
    with additional type checking and JSON schema generation.
    """

    spiketrain: Any = Field(..., description="Spiketrain Object")
    dither: Any = Field(..., description="maximum displacement of a spike")
    n_surrogates: Optional[int] = Field(1, gt=0, description="Number of generated surrogates")


    @field_validator("spiketrain")
    @classmethod
    def validate_spiktetrain(cls, value, info):
        return fv.validate_spiketrain(value, info, allowed_types=(neo.SpikeTrain,))
    
    @field_validator("dither")
    @classmethod
    def validate_quantity(cls, v, info):
        return fv.validate_quantity(v, info)
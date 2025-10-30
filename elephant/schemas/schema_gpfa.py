import quantities as pq
import numpy as np
from typing import (
    Any,
    List,
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
import elephant
from elephant.schemas.class_builder import make_class_model
import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs


class GPFAInit(BaseModel):
    """
    Constructor parameters for elephant.gpfa.gpfa.GPFA
    """
    bin_size: Optional[Any] = Field(default_factory= lambda: 20 * pq.ms, description="spike bin width")
    x_dim: Optional[int] = Field(3, ge=1, description="state dimensionality")
    min_var_frac: Optional[float] = Field(0.01, ge=0.0, description="minimum private variance fraction")
    tau_init: Optional[Any] = Field(default_factory=lambda: 100.0 * pq.ms, description="GP timescale initialization")
    eps_init: Optional[float] = Field(1.0e-3, ge=0.0, description="GP noise variance initialization")
    em_tol: Optional[float] = Field(1.0e-8, gt=0.0, description="EM stopping tolerance")
    em_max_iters: Optional[int] = Field(500, ge=1, description="Maximum EM iterations")
    freq_ll: Optional[int] = Field(5, ge=1, description="Compute data likelihood every freq_ll EM iterations")
    verbose: Optional[bool] = Field(False, description="Display status messages")

    @field_serializer("bin_size", "tau_init", mode='plain')
    def serialize_bin_size(self, v):
        return fs.serialize_quantity(v)

    @field_validator("bin_size", "tau_init")
    @classmethod
    def validate_quantities(cls, value, info):
        return fv.validate_quantity(value, info)


class PydanticGPFAFit(BaseModel):
    """
    Pydantic wrapper for `elephant.gpfa.gpfa.GPFA.fit`
    """

    spiketrains: Any = Field(..., description="Spike train data to be fit to latent variables")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains_matrix(v, info, check_rank_deficient=True)


class PydanticGPFATransform(BaseModel):
    """
    Pydantic wrapper for `elephant.gpfa.gpfa.GPFA.transform`
    """

    class ReturnedDataOptions(Enum):
        latent_variable_orth = "latent_variable_orth"
        latent_variable = "latent_variable"
        Vsm = "Vsm"
        VsmGP = "VsmGP"
        y = "y"

    spiketrains: list = Field(..., description="Spike train data to be transformed to latent variables")
    returned_data: list[ReturnedDataOptions] = Field(["latent_variable_orth"], description="Keys of data to return")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains_matrix(v, info)


class PydanticGPFAFitTransform(BaseModel):
    """
    Pydantic wrapper for `elephant.gpfa.gpfa.GPFA.fit_transform`
    """

    class ReturnedDataOptions(Enum):
        latent_variable_orth = "latent_variable_orth"
        latent_variable = "latent_variable"
        Vsm = "Vsm"
        VsmGP = "VsmGP"
        y = "y"

    spiketrains: list = Field(..., description="Spike train data to be fit and transformed")
    returned_data: list[ReturnedDataOptions] = Field(["latent_variable_orth"], description="Keys of data to return")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains_matrix(v, info, check_rank_deficient=True)


class PydanticGPFAScore(BaseModel):
    """
    Pydantic wrapper for `elephant.gpfa.gpfa.GPFA.score`
    """

    spiketrains: list = Field(..., description="Spike train data to be scored")

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains_matrix(v, info)


PydanticGPFAModel = make_class_model(
    "GPFA",
    {
        "constructor": GPFAInit,
        "fit": PydanticGPFAFit,
        "transform": PydanticGPFATransform,
        "fit_transform": PydanticGPFAFitTransform,
        "score": PydanticGPFAScore,
    },
)
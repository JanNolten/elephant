import quantities as pq
import numpy as np
from typing import (
    Any,
    Union,
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
from elephant.schemas.class_builder import make_class_model
from os import PathLike
import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

import warnings


class PydanticSynchronousEventsIntersection(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_intersection function
    with additional type checking and json_schema by PyDantic.
    """

    class IntersectionOptions(Enum):
        pixelwise = "pixelwise"
        linkwise = "linkwise"

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    intersection: Optional[IntersectionOptions] = Field(IntersectionOptions.linkwise, description="Type of intersection to perform")

class PydanticSynchronousEventsDifference(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_difference function
    with additional type checking and json_schema by PyDantic.
    """

    class DifferenceOptions(Enum):
        pixelwise = "pixelwise"
        linkwise = "linkwise"

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    difference: Optional[DifferenceOptions] = Field(DifferenceOptions.linkwise, description="Type of difference to perform")

class PydanticSynchronousEventsIdentical(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_identical function
    with additional type checking and json_schema by PyDantic.
    """

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticSynchronousEventsNoOverlap(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_no_overlap function
    with additional type checking and json_schema by PyDantic.
    """

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticSynchronousEventsContainedIn(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_contained_in function
    with additional type checking and json_schema by PyDantic.
    """

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticSynchronousEventsContainsAll(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_contains_all function
    with additional type checking and json_schema by PyDantic.
    """

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticSynchronousEventsOverlap(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.synchronous_events_overlap function
    with additional type checking and json_schema by PyDantic.
    """

    sse1: dict = Field(..., description="Dictionary of pixel postion as keys and sets")
    sse2: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticGetNeuronsInSse(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.get_neurons_in_sse function
    with additional type checking and json_schema by PyDantic.
    """

    sse: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticGetsseStartAndEndTimeBins(BaseModel):
    """
    PyDantic Class to wrap the elephant.asset.asset.get_sse_start_and_end_time_bins function
    with additional type checking and json_schema by PyDantic.
    """

    sse: dict = Field(..., description="Dictionary of pixel postion as keys and sets")

class PydanticASSETInit(BaseModel):
    class BinToleranceOptions(Enum):
        _default = "default"

    spiketrains_i: list = Field(..., description="List of neo.SpikeTrain objects")
    spiketrains_j: list = Field(None, description="List of neo.SpikeTrain objects")
    bin_size: Optional[Any] = Field(None, description="Width of time bins")
    t_start_i: Optional[Any] = Field(None, description="Start time")
    t_stop_i: Optional[Any] = Field(None, description="Stop time")
    t_start_j: Optional[Any] = Field(None, description="Start time")
    t_stop_j: Optional[Any] = Field(None, description="Stop time")
    bin_tolerance: Optional[Union[float, BinToleranceOptions]] = Field(BinToleranceOptions._default, description="Tolerance for rounding errors")

    @field_validator("spiketrains_i")
    @classmethod
    def validate_spiketrains_i(cls, v, info):
        return fv.validate_spiketrains(v, info, allowed_content_types=(neo.SpikeTrain,))
    
    @field_validator("spiketrains_j")
    @classmethod
    def validate_spiketrains_j(cls, v, info):
        return fv.validate_spiketrains(v, info, allow_none=True, allowed_content_types=(neo.SpikeTrain,))

    @field_validator("bin_size")
    @classmethod
    def validate_bin_size(cls, v, info):
        return fv.validate_quantity(v, info)

    @field_validator("t_start_i", "t_stop_i", "t_start_j", "t_stop_j")
    @classmethod
    def validate_time(cls, v, info):
        return fv.validate_time(v, info)
    
class PydanticASSETClusterMatrixEntries(BaseModel):
    mask_matrix: Any = Field(..., description="To cluster boolean matrix")
    max_distance: float = Field(..., ge=0, description="Maximum distance between two elements")
    min_neighbours: int = Field(..., ge=0, description="Minimum number of elements for neighbourhood")
    stretch: float = Field(..., gt=0, description="Stretching factor")
    working_memory: Optional[int] = Field(None, ge=0, description="Sought maximum memory in MiB")
    array_file: Optional[Union[str, PathLike]] = Field(None, description="Path temporary store file")
    keep_file: Optional[bool] = Field(False, description="Keep temporary file")

    @field_validator("mask_matrix")
    @classmethod
    def validate_matrix(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,), allow_none=False)
    
class PydanticASSETExtractSynchronousEvents(BaseModel):
    cmat: Any = Field(..., description="Cluster matrix")
    ids: Optional[list] = Field(None, description="List of spike train IDs")

    @field_validator("cmat")
    @classmethod
    def validate_matrix(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,), allow_none=False)
    
class PydanticASSETIntersectionMatrix(BaseModel):
    class NormalizationOptions(Enum):
        intersection = "intersection"
        mean = "mean"
        union = "union"

    normalization: Optional[NormalizationOptions] = Field(None, description="Normalization type for intersection matrix")

class PydanticASSETJointProbabilityMatrix(BaseModel):
    class PrecisionOptions(Enum):
        _float = "float"
        _double = "double"

    pmat: Any = Field(..., description="Probability matrix")
    filter_shape: tuple[int, int] = Field(..., description="Kernel shape (l, w)")
    n_largest: int = Field(..., gt=0, description="Number of largest neighbors to collect")
    min_p_value: Optional[float] = Field(1e-5, ge=0, le=1, description="Minimum individual p-value")
    precision: Optional[PrecisionOptions] = Field(PrecisionOptions._float, description="Precision mode")
    cuda_threads: Optional[int] = Field(64, ge=1, le=1024, description="CUDA threads per block (if GPU enabled)")
    cuda_cwr_loops: Optional[int] = Field(32, ge=1, description="CUDA combination loops")
    tolerance: Optional[float] = Field(1e-5, ge=0, description="Tolerance for floating-point errors")

    @field_validator("pmat")
    @classmethod
    def validate_matrix(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,), allow_none=False)
    
    @field_validator("cuda_threads")
    @classmethod
    def validate_cuda_threads(cls, v):
        if v % 32 != 0:
            warnings.warn("cuda_threads should be a multiple of 32", UserWarning)
        return v

class PydanticASSETMaskMatrices(BaseModel):
    matrices: list= Field(..., description="List of matrices to compare")
    thresholds: Union[float, list[float]] = Field(..., description="Threshold(s) per matrix")

    @field_validator("matrices")
    @classmethod
    def validate_matrices(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(list,), allowed_content_types=(np.ndarray,))
    
    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v, info):
        # thresholds can be a single float or a list of floats
        if isinstance(v, list):
            return fv.validate_array(v, info, allowed_types=(list,), allowed_content_types=(float,))
        if isinstance(v, float):
            return v
        raise TypeError(f"{info.field_name} must be a float or list of floats")
    
    @model_validator(mode="after")
    def check_correctTypeCombination(self):             
        if(isinstance(self.thresholds, list) and len(self.matrices) != len(self.thresholds)):
            raise ValueError("matrices and thresholds need to have the same length")
        return self

class PydanticASSETProbabilityMatrixAnalytical(BaseModel):
    class FiringRatesOptions(Enum):
        estimate = "estimate"

    imat: Optional[Any] = Field(None, description="Intersection matrix")
    firing_rates_x: Union[FiringRatesOptions, list] = Field(FiringRatesOptions.estimate, description="Firing rates for X spike trains")
    firing_rates_y: Union[FiringRatesOptions, list] = Field(FiringRatesOptions.estimate, description="Firing rates for Y spike trains")
    kernel_width: Optional[Any] = Field(default_factory=100 * pq.ms, description="Kernel width for rate estimation")

    @field_serializer("kernel_width", mode='plain')
    def serialize_quantity(self, value: pq.Quantity):
        return fs.serialize_quantity(value)

    @field_validator("imat")
    @classmethod
    def validate_matrix(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(np.ndarray,), allow_none=True)
    
    @field_validator("firing_rates_x", "firing_rates_y")
    @classmethod
    def validate_matrices(cls, v, info):
        if not isinstance(v, list):
            return v
        return fv.validate_array(v, info, allowed_types=(list,), allowed_content_types=(neo.AnalogSignal,))
    
    @field_validator("kernel_width")
    @classmethod
    def validate_kernel_width(cls, v, info):
        return fv.validate_quantity(v, info)
    
class PydanticASSETProbabilityMatrixMonteCarlo(BaseModel):
    class SurrogateMethodOptions(Enum):
        dither_spike_train = "dither_spike_train"
        dither_spikes = "dither_spikes"
        jitter_spikes = "jitter_spikes"
        randomise_spikes = "randomise_spikes"
        shuffle_isis = "shuffle_isis"
        joint_isi_dithering = "joint_isi_dithering"

    n_surrogates: int = Field(..., description="Number of surrogate datasets to generate")
    imat: Optional[Any] = Field(None, description="Intersection matrix")
    surrogate_method: Optional[SurrogateMethodOptions] = Field(SurrogateMethodOptions.dither_spike_train, description="Surrogate generation method (see spike_train_surrogates.surrogates)")
    surrogate_dt: Optional[Any] = Field(None, description="Shift/jitter window for surrogate methods")

    @field_validator("imat")
    @classmethod
    def validate_matrix(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            raise TypeError("imat must be a numpy.ndarray")
        return v
    
    @field_validator("surrogate_dt")
    @classmethod
    def validate_surrogate_dt(cls, v, info):
        return fv.validate_quantity(v, info, allow_none=True)
    
PydanticASSET = make_class_model(
    "ASSET",
    {
        "constructor": PydanticASSETInit,
        "cluster_matrix_entries": PydanticASSETClusterMatrixEntries,
        "extract_synchronous_events": PydanticASSETExtractSynchronousEvents,
        "intersection_matrix": PydanticASSETIntersectionMatrix,
        "is_symmetric": None,
        "joint_probability_matrix": PydanticASSETJointProbabilityMatrix,
        "mask_matrices": PydanticASSETMaskMatrices,
        "probability_matrix_analytical": PydanticASSETProbabilityMatrixAnalytical,
        "probability_matrix_montecarlo": PydanticASSETProbabilityMatrixMonteCarlo,
    }
)
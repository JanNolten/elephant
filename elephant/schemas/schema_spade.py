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
    field_serializer
)
import neo
from enum import Enum
import elephant
import scipy.sparse as sp

import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

class StatCorrOptions(str, Enum):
    bonferroni = "bonferroni"
    sidak = "sidak"
    holm_sidak = "holm-sidak"
    holm = "holm"
    simes_hochberg = "simes-hochberg"
    hommel = "hommel"
    fdr_bh = "fdr_bh"
    fdr_by = "fdr_by"
    fdr_tsbh = "fdr_tsbh"
    fdr_tsbky = "fdr_tsbky"
    no = "no"


#Should get from elephant.spike_train_surrogates.surrogates
class SurrogateMethodOptions(str, Enum):
    dither_spikes = "dither_spikes"
    dither_spike_train = "dither_spike_train"
    jitter_spikes = "jitter_spikes"
    randomise_spikes = "randomise_spikes"
    shuffle_isis = "shuffle_isis"
    joint_isi_dithering = "joint_isi_dithering"
    trial_shifting = "trial_shifting"
    bin_shuffling = "bin_shuffling"

class PydanticSpade(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spade.spade` function
    with additional type checking and JSON schema generation.

    Perform the SPADE analysis for parallel spike trains using Frequent Itemset Mining (FIM)
    or Formal Concept Analysis (FCA) to detect repeating spatio-temporal spike patterns.
    """

    class ApproxStabParsOptions(str, Enum):
        n_subsets = "n_subsets"
        delta = "delta"
        epsilon = "epsilon"
        stability_thresh = "stability_thresh"

    class SpectrumOptions(str, Enum):
        hash = "#"
        three_d_hash = "3d#"

    class OutputFormatOptions(str, Enum):
        concepts = "concepts"
        patterns = "patterns"


    spiketrains: list = Field(..., description="List of neo.core.SpikeTrain objects")
    bin_size: Any = Field(..., description="Time precision to discretize spiketrains")
    winlen: int = Field(..., gt=0, description="Number of bins per sliding analysis window")
    min_spikes: Optional[int] = Field(2, ge=1, description="Min number of spikes in pattern")
    min_occ: Optional[int] = Field(2, ge=1, description="Min number of occurrences of pattern")
    max_spikes: Optional[int] = Field(None, ge=1, description="Max number of spikes in pattern")
    max_occ: Optional[int] = Field(None, ge=1, description="Max number of occurrences of pattern")
    min_neu: Optional[int] = Field(1, ge=1, description="Min number of neurons required in pattern")
    approx_stab_pars: Optional[dict[ApproxStabParsOptions, Union[None,int,float,list[float]]]] = Field(None,description=("Parameter values for approximate statbility computation"))
    n_surr: Optional[int] = Field(0, ge=0, description="Number of surrogates for p-value computation")
    dither: Optional[Any] = Field(default_factory=lambda: 15 * pq.ms, ge=0, description="Spike time dithering window for surrogates")
    spectrum: Optional[SpectrumOptions] = Field(
        SpectrumOptions.hash,
        description="Pattern spectrum signature"
    )
    alpha: Optional[float] = Field(None, gt=0, le=1, description="Significance level for hypothesis tests")
    stat_corr: Optional[StatCorrOptions] = Field(
        StatCorrOptions.fdr_bh,
        description="Multiple testing correction method"
    )
    surr_method: Optional[SurrogateMethodOptions] = Field(
        SurrogateMethodOptions.dither_spikes,
        description="Surrogate generation method"
    )
    psr_param: Optional[Union[list[int], tuple[int,]]] = Field(
        None,
        description=(
            "Pattern Spectrum Reduction parameters"
        )
    )
    output_format: Optional[OutputFormatOptions] = Field(
        OutputFormatOptions.patterns,
        description="Output format"
    )
    surr_kwargs: Optional[dict[str, Any]] = Field(
        None,
        description="Keyword arguments for surrogate methods"
    )

    @field_serializer("dither", mode='plain')
    def serialize_quantity(self, v):
        return fs.serialize_quantity(v)

    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains(v, info, allowed_content_types=(neo.core.SpikeTrain,))
    
    @field_validator("bin_size", "dither")
    @classmethod
    def validate_bin_size(cls, value, info):
        return fv.validate_quantity(value, info)
    
    @field_validator("approx_stab_pars")
    @classmethod
    def validate_approx_stab_pars(cls, value, info):
        if value is not None:
            expected_types = {
                cls.ApproxStabParsOptions.n_subsets: int,
                cls.ApproxStabParsOptions.delta: float,
                cls.ApproxStabParsOptions.epsilon: float,
                cls.ApproxStabParsOptions.stability_thresh: (None, float, list)
            }
            fv.validate_dict_enum_types(value, info, expected_types)
        return value

class PydanticConceptsMining(BaseModel):
    """
    Pydantic wrapper for elephant.spade.concepts_mining

    Find pattern candidates extracting all the concepts of the context, formed by
    the objects defined as all windows of length winlen*bin_size slided along
    the discretized spiketrains and the attributes as the spikes occurring in
    each of the windows.
    """

    class ReportOptions(str, Enum):
        a = "a"
        hash = "#"
        three_d_hash = "3d#"

    spiketrains: list = Field(..., description="List of neo.core.SpikeTrain objects or a BinnedSpikeTrain")
    bin_size: Any = Field(..., description="Time precision to discretize spiketrains")
    winlen: int = Field(..., description="Number of bins per sliding analysis window")
    min_spikes: Optional[int] = Field(2, description="Min number of spikes in pattern")
    min_occ: Optional[int] = Field(2, description="Min number of occurrences of pattern")
    max_spikes: Optional[int] = Field(None, description="Max number of spikes in pattern")
    max_occ: Optional[int] = Field(None, description="Max number of occurrences of pattern")
    min_neu: Optional[int] = Field(1, description="Min number of neurons required in pattern")
    report: Optional[ReportOptions] = Field(
        ReportOptions.a,
        description=(
            "Indicates output of function"
        )
    )

    # Validators
    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains(v, info, allowed_content_types=(neo.core.SpikeTrain, elephant.conversion.BinnedSpikeTrain))

    @field_validator("bin_size")
    @classmethod
    def validate_bin_size(cls, value, info):
        return fv.validate_quantity(value, info)


class PydanticPValueSpectrum(BaseModel):
    """
    Pydantic wrapper for elephant.spade.pvalue_spectrum

    Compute the p-value spectrum of pattern signatures extracted from surrogates
    of parallel spike trains, under the null hypothesis of independent spiking.
    """

    class SpectrumOptionsLocal(str, Enum):
        hash = "#"
        three_d_hash = "3d#"

    spiketrains: list = Field(..., description="List of neo.core.SpikeTrain objects")
    bin_size: Any = Field(..., description="Time precision to discretize spiketrains")
    winlen: int = Field(..., description="Number of bins per sliding analysis window")
    dither: Any = Field(default_factory=lambda: 15 * pq.ms, description="Spike time dithering window for surrogates")
    n_surr: int = Field(0, ge=0, description="Number of surrogates to generate for p-value spectrum")
    min_spikes: Optional[int] = Field(2, description="Min number of spikes in pattern")
    min_occ: Optional[int] = Field(2, description="Min number of occurrences of pattern")
    max_spikes: Optional[int] = Field(None, description="Max number of spikes in pattern")
    max_occ: Optional[int] = Field(None, description="Max number of occurrences of pattern")
    min_neu: Optional[int] = Field(1, description="Min number of neurons required in pattern")
    spectrum: Optional[SpectrumOptionsLocal] = Field(
        SpectrumOptionsLocal.hash,
        description="Pattern spectrum signature"
    )
    surr_method: Optional[SurrogateMethodOptions] = Field(
        SurrogateMethodOptions.dither_spikes,
        description="Surrogate generation method"
    )
    surr_kwargs: Optional[dict[str, Any]] = Field(
        None,
        description="Keyword arguments for surrogate methods"
    )
    
    @field_serializer("dither", mode='plain')
    def serialize_quantity(self, v):
        return fs.serialize_quantity(v)

    # Validators
    @field_validator("spiketrains")
    @classmethod
    def validate_spiketrains(cls, v, info):
        return fv.validate_spiketrains(v, info, allowed_content_types=(neo.core.SpikeTrain,))

    @field_validator("bin_size", "dither")
    @classmethod
    def validate_quantities(cls, value, info):
        return fv.validate_quantity(value, info)
    


class PydanticTestSignatureSignificance(BaseModel):
    """
    Pydantic wrapper for elephant.spade.test_signature_significance
    """

    class ReportOptionsLocal(str, Enum):
        spectrum = "spectrum"
        significant = "significant"
        non_significant = "non_significant"

    class SpectrumOptionsLocal(str, Enum):
        hash = "#"
        three_d_hash = "3d#"

    pv_spec: list[tuple] = Field(..., description="p-value spectrum as list of signatures with p-values")
    concepts: list[tuple] = Field(..., description="Concepts mined from original data")
    alpha: float = Field(..., ge=0, le=1, description="Significance level for the statistical test")
    winlen: int = Field(..., gt=0, description="Number of bins per sliding analysis window")
    corr: Optional[StatCorrOptions] = Field(StatCorrOptions.fdr_bh, description="Multiple testing correction method")
    report: Optional[ReportOptionsLocal] = Field(ReportOptionsLocal.spectrum, description="Format of returned significance spectrum")
    spectrum: Optional[SpectrumOptionsLocal] = Field(SpectrumOptionsLocal.hash, description="Pattern spectrum signature")

    # Validators
    @field_validator("pv_spec")
    @classmethod
    def validate_pv_spec(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(list,), allow_none=False, min_length=1, allowed_content_types=(tuple), min_length_content=3)


class PydanticApproximateStability(BaseModel):
    """
    Pydantic wrapper for elephant.spade.approximate_stability
    """

    concepts: list[tuple] = Field(..., description="All pattern candidates (concepts)")
    rel_matrix: Any = Field(..., description="Relation matrix (binary) as dense or sparse array")
    n_subsets: int = Field(0, ge=0, description="Number of subsets for approximation")
    delta: Optional[float] = Field(0., ge=0., le=1., description="Delta parameter for approximation")
    epsilon: Optional[float] = Field(0., ge=0., description="Epsilon parameter for approximation")

    @field_validator("concepts")
    @classmethod
    def validate_concepts(cls, v, info):
        return fv.validate_array(v, info, allowed_types=(list,), allow_none=False, min_length=0, allowed_content_types=(tuple,), min_length_content=2)

    @field_validator("rel_matrix")
    @classmethod
    def validate_rel_matrix(cls, v, info):
        return fv.validate_type(v, info, (sp.coo_matrix,), allow_none=False)


class PydanticPatternSetReduction(BaseModel):
    """
    Pydantic wrapper for elephant.spade.pattern_set_reduction
    """

    class SpectrumOptions(str, Enum):
        hash = "#"
        three_d_hash = "3d#"

    concepts: list = Field(..., description="List of concepts to reduce")
    ns_signatures: list[tuple] = Field(..., description="List of non-significant signatures")
    winlen: int = Field(..., gt=0, description="Number of bins per sliding analysis window")
    spectrum: SpectrumOptions = Field(..., description="Pattern spectrum signature")
    h_subset_filtering: Optional[int] = Field(0, description="Correction parameter for subset filtering")
    k_superset_filtering: Optional[int] = Field(0, description="Correction parameter for superset filtering")
    l_covered_spikes: Optional[int] = Field(0, description="Correction parameter for covered-spikes criterion")
    min_spikes: Optional[int] = Field(2, gt=0, description="Minimum pattern size")
    min_occ: Optional[int] = Field(2, gt=0, description="Minimum number of pattern occurrences")


class PydanticConceptOutputToPatterns(BaseModel):
    """
    Pydantic wrapper for elephant.spade.concept_output_to_patterns
    """

    class SpectrumOptions(str, Enum):
        hash = "#"
        three_d_hash = "3d#"

    concepts: tuple = Field(..., description="Tuple of concepts (intent, extent)")
    winlen: int = Field(..., gt=0, description="Length (in bins) of the sliding window")
    bin_size: Any = Field(..., description="Time precision used to discretize the spiketrains")
    pv_spec: Union[None, tuple] = Field(None, description="p-value spectrum")
    spectrum: SpectrumOptions = Field(SpectrumOptions.hash, description="Pattern spectrum signature")
    t_start: Any = Field(default_factory=lambda: 0 * pq.ms, description="t_start of analyzed spike trains")

    @field_serializer("t_start", mode='plain')
    def serialize_quantity(self, v):
        return fs.serialize_quantity(v)
    
    @field_validator("bin_size", "t_start")
    @classmethod
    def validate_quantities(cls, value, info):
        return fv.validate_quantity(value, info)


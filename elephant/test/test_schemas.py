
import pytest
import quantities as pq
import neo
import numpy as np

import elephant

from elephant.schemas.function_validator import deactivate_validation

from elephant.schemas.schema_asset import *
from elephant.schemas.schema_cell_assembly_detection import *
from elephant.schemas.schema_change_point_detection import *
from elephant.schemas.schema_cubic import *
from elephant.schemas.schema_functional_connectivity import *
from elephant.schemas.schema_spade import *
from elephant.schemas.schema_unitary_event_analysis import *
from elephant.schemas.schema_statistics import *
from elephant.schemas.schema_spike_train_correlation import *
from elephant.schemas.schema_spike_train_dissimilarity import *
from elephant.schemas.schema_spike_train_synchrony import *
from elephant.schemas.schema_gpfa import *
from elephant.schemas.schema_spike_train_surrogates import *


def test_model_json_schema():
	# Just test that json_schema generation runs without error for all models
	model_classes = [
		PydanticSynchronousEventsIntersection,
		PydanticSynchronousEventsDifference,
		PydanticSynchronousEventsIdentical,
		PydanticSynchronousEventsNoOverlap,
		PydanticSynchronousEventsContainedIn,
		PydanticSynchronousEventsContainsAll,
		PydanticSynchronousEventsOverlap,
		PydanticGetNeuronsInSse,
		PydanticGetsseStartAndEndTimeBins,
		PydanticCellAssemblyDetection,
		PydanticCovariance,
		PydanticCorrelationCoefficient,
		PydanticCrossCorrelationHistogram,
		PydanticSpikeTimeTilingCoefficient,
		PydanticSpikeTrainTimescale,
		PydanticSpade,
		PydanticConceptsMining,
		PydanticPValueSpectrum,
		PydanticTestSignatureSignificance,
		PydanticApproximateStability,
		PydanticPatternSetReduction,
		PydanticConceptOutputToPatterns,
		PydanticVictorPurpuraDistance,
		PydanticVanRossumDistance,
		PydanticMeanFiringRate,
		PydanticInstantaneousRate,
		PydanticTimeHistogram,
		PydanticOptimalKernelBandwidth,
		PydanticIsi,
		PydanticCv,
		PydanticCv2,
		PydanticLv,
		PydanticLvr,
		PydanticFanofactor,
		PydanticComplexityPdf,
		PydanticSpikeContrast,
		PydanticJointJWindowAnalysis,
		PydanticCubic,
		PydanticMultipleFilterTest,
		PydanticEmpiricalParameters,
		PydanticTotalSpikingProbabilityEdges,
		PydanticASSET,
		PydanticSynchrotool,
		PydanticComplexity,
		PydanticGPFA,
		PydanticSurrogates,
		PydanticJointISI,
		PydanticDitherSpikes,
		PydanticRandomiseSpikes,
		PydanticShuffleIsis,
		PydanticDitherSpikeTrain,
		PydanticJitterSpikes,
		PydanticBinShuffling,
		PydanticTrialShifting,
	]
	for cls in model_classes:
		schema = cls.model_json_schema()
		assert isinstance(schema, dict)

# Deactivate validation happening in the decorator of the elephant functions for all tests in this module to keep checking consistent behavior
@pytest.fixture(autouse=True)
def disable_validation_for_tests():
	deactivate_validation()

@pytest.fixture
def make_list():
	return [0.01, 0.02, 0.05]

@pytest.fixture
def make_ndarray(make_list):
	return np.array(make_list)

@pytest.fixture
def make_pq_single_quantity():
	return 0.05 * pq.s

@pytest.fixture
def make_pq_multiple_quantity(make_ndarray):
	return make_ndarray * pq.s

@pytest.fixture
def make_spiketrain(make_pq_multiple_quantity):
	return neo.core.SpikeTrain(make_pq_multiple_quantity, t_start=0 * pq.s, t_stop=0.1 * pq.s)

@pytest.fixture
def make_spiketrains(make_spiketrain):
	return [make_spiketrain, make_spiketrain]

@pytest.fixture
def make_binned_spiketrain(make_spiketrain):
	return elephant.conversion.BinnedSpikeTrain(make_spiketrain, bin_size=0.01 * pq.s)

@pytest.fixture
def make_analog_signal():
	n2 = 300
	n0 = 100000 - n2
	return neo.AnalogSignal(np.array([10] * n2 + [0] * n0).reshape(n0 + n2, 1) * pq.dimensionless, sampling_period=1 * pq.s)

@pytest.fixture
def fixture(request):
	return request.getfixturevalue(request.param)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.mean_firing_rate, PydanticMeanFiringRate),
	(elephant.statistics.isi, PydanticIsi),
])
@pytest.mark.parametrize("fixture", [
	"make_list",
	"make_spiketrain",
    "make_ndarray",
    "make_pq_multiple_quantity",
], indirect=["fixture"])
def test_valid_spiketrain_input(elephant_fn, model_cls, fixture):
	valid = {"spiketrain": fixture}
	assert(isinstance(model_cls(**valid), model_cls))
	# just check it runs without error
	elephant_fn(**valid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.mean_firing_rate, PydanticMeanFiringRate),
	(elephant.statistics.isi, PydanticIsi),
])
@pytest.mark.parametrize("spiketrain", [
	5,
	"hello",
])
def test_invalid_spiketrain(elephant_fn, model_cls, spiketrain):
	invalid = {"spiketrain": spiketrain}
	with pytest.raises(Exception):
		model_cls(**invalid)
	with pytest.raises(Exception):
		elephant_fn(**invalid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.time_histogram, PydanticTimeHistogram),
	(elephant.statistics.complexity_pdf, PydanticComplexityPdf),
])
def test_valid_pq_quantity(elephant_fn, model_cls, make_spiketrains, make_pq_single_quantity):
	valid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity}
	assert(isinstance(model_cls(**valid), model_cls))
	# just check it runs without error
	elephant_fn(**valid)


@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.time_histogram, PydanticTimeHistogram),
	(elephant.statistics.complexity_pdf, PydanticComplexityPdf),
])
@pytest.mark.parametrize("pq_quantity", [
	5,
	"hello",
	[0.01, 0.02]
])
def test_invalid_pq_quantity(elephant_fn, model_cls, make_spiketrains, pq_quantity):
	invalid = {"spiketrains": make_spiketrains, "bin_size": pq_quantity}
	with pytest.raises(Exception):
		model_cls(**invalid)
	with pytest.raises(Exception):
		elephant_fn(**invalid)



@pytest.mark.parametrize("elephant_fn,model_cls", [
	(elephant.statistics.instantaneous_rate, PydanticInstantaneousRate),
])
@pytest.mark.parametrize("fixture", [
	"make_list",
    "make_ndarray",
    "make_pq_multiple_quantity",
], indirect=["fixture"])
def test_invalid_spiketrains(elephant_fn, model_cls, fixture, make_pq_single_quantity):
	invalid = {"spiketrains": fixture, "sampling_period": make_pq_single_quantity}
	with pytest.raises(Exception):
		model_cls(**invalid)
	with pytest.raises(Exception):
		elephant_fn(**invalid)

@pytest.mark.parametrize("output", [
	"counts",
	"mean",
	"rate",
])
def test_valid_enum(output, make_spiketrains, make_pq_single_quantity):
	valid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "output": output}
	assert(isinstance(PydanticTimeHistogram(**valid), PydanticTimeHistogram))
	# just check it runs without error
	elephant.statistics.time_histogram(**valid)

@pytest.mark.parametrize("output", [
	"countsfagre",
	5,
	"Counts",
	"counts ",
	" counts",
	"counts\n"
])
def test_invalid_enum(output, make_spiketrains, make_pq_single_quantity):
	invalid = {"spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "output": output}
	with pytest.raises(Exception):
		PydanticTimeHistogram(**invalid)
	with pytest.raises(Exception):
		elephant.statistics.time_histogram(**invalid)


def test_valid_binned_spiketrain(make_binned_spiketrain):
	valid = {"binned_spiketrain": make_binned_spiketrain}
	assert(isinstance(PydanticCovariance(**valid), PydanticCovariance))
	# just check it runs without error
	elephant.spike_train_correlation.covariance(**valid)

def test_invalid_binned_spiketrain(make_spiketrain):
	invalid = {"binned_spiketrain": make_spiketrain}
	with pytest.raises(Exception):
		PydanticCovariance(**invalid)
	with pytest.raises(Exception):
		elephant.spike_train_correlation.covariance(**invalid)

@pytest.mark.parametrize("elephant_fn,model_cls,parameter_name,empty_input", [
	(elephant.statistics.instantaneous_rate, PydanticInstantaneousRate, "spiketrains", []),
	(elephant.statistics.optimal_kernel_bandwidth, PydanticOptimalKernelBandwidth, "spiketimes", np.array([])),
	(elephant.statistics.cv2, PydanticCv2, "time_intervals", np.array([])*pq.s),
])
def test_invalid_empty_input(elephant_fn, model_cls, parameter_name, empty_input):
	invalid = {parameter_name: empty_input}
	with pytest.raises(Exception):
		model_cls(**invalid)
	with pytest.raises(Exception):
		elephant_fn(**invalid)

@pytest.mark.parametrize("elephant_fn,model_cls,parameter_name,empty_input", [
	(elephant.spike_train_correlation.covariance, PydanticCovariance, "binned_spiketrain", elephant.conversion.BinnedSpikeTrain(neo.core.SpikeTrain(np.array([])*pq.s, t_start=0*pq.s, t_stop=1*pq.s), bin_size=0.01*pq.s)),
])
def test_warning_empty_input(elephant_fn, model_cls, parameter_name, empty_input):
	warning = {parameter_name: empty_input}
	with pytest.warns(Warning):
		model_cls(**warning)
	with pytest.warns(Warning):
		elephant_fn(**warning)


def test_valid_Complexity(make_spiketrains, make_pq_single_quantity):
	valid = { "spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity }
	assert(isinstance(PydanticComplexityInit(**valid), PydanticComplexityInit))
	# just check it runs without error
	elephant.statistics.Complexity(**valid)


def test_valid_dynamic_enum(make_spiketrains, make_pq_single_quantity, make_spiketrain):
	valid = { "spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "winlen": 1, "dither": 15*pq.s, "n_surr": 1, "surr_method": "bin_shuffling", "sliding": True}
	assert(isinstance(PydanticPValueSpectrum(**valid), PydanticPValueSpectrum))
	# just check it runs without error
	elephant.statistics.pvalue_spectrum(**valid)

@pytest.mark.parametrize("surr_method", [
	"JointISI",
	5,
	"Randomise_spikes",
	"randomise_spikes ",
])
def test_invalid_dynamic_enum(make_spiketrains, make_pq_single_quantity, surr_method):
	valid = { "spiketrains": make_spiketrains, "bin_size": make_pq_single_quantity, "winlen": 1, "dither": 15*pq.s, "n_surr": 1, "surr_method": surr_method}
	with pytest.raises(Exception):
		PydanticPValueSpectrum(**valid)
	with pytest.raises(Exception):
		elephant.spade.pvalue_spectrum(**valid)


def test_valid_analog_signal(make_analog_signal):
	valid = { "histogram": make_analog_signal }
	assert(isinstance(PydanticCubic(**valid), PydanticCubic))
	# just check it runs without error
	elephant.cubic.cubic(**valid)


@pytest.mark.parametrize("fixture", [
	"make_list",
    "make_ndarray",
    "make_pq_multiple_quantity",
], indirect=["fixture"])
def test_invalid_analog_signal(fixture):
	invalid = { "histogram": fixture}
	with pytest.raises(Exception):
		PydanticCubic(**invalid)
	with pytest.raises(Exception):
		elephant.cubic.cubic(**invalid)
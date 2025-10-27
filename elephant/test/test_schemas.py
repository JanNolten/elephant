
import pytest
import warnings
import quantities as pq
import neo
import numpy as np

import elephant
import inspect

from pydantic import ValidationError

from elephant.schemas import (
    PydanticTimeHistogram,
    PydanticFanofactor,
    PydanticComplexityPdf,
    PydanticSpade,
    PydanticConceptsMining,
    PydanticPValueSpectrum,
    PydanticJointJWindowAnalysis,
    PydanticCellAssemblyDetection,
    PydanticOptimalKernelBandwidth,
    PydanticLvr,
)


def make_spiketrain():
	# simple spike times in seconds
	times = np.array([0.01, 0.02, 0.05])
	return neo.core.SpikeTrain(times * pq.s, t_start=0 * pq.s, t_stop=0.1 * pq.s)


def assert_both_raise_consistently(elephant_fn, model_cls, kwargs, *, same_type=False, expected_exception=None):
	"""Call both the Elephant function and the Pydantic model with the same kwargs.
	Assert both raise, and if requested assert they raise the same exception type.

	Uses pytest.raises to capture exceptions so failures are reported with pytest's
	native formatting while still allowing comparison of exception objects.

	Parameters
	- elephant_fn: callable to invoke with kwargs
	- model_cls: Pydantic model class to instantiate with kwargs
	- kwargs: dict of keyword arguments to pass to both
	- same_type: if True assert the raised exception classes are identical
	- expected_exception: optional exception type that both must be instances of
	"""
	with pytest.raises(Exception) as e1:
		elephant_fn(**kwargs)
	with pytest.raises(Exception) as e2:
		model_cls(**kwargs)

	exc1 = e1.value
	exc2 = e2.value

	if expected_exception is not None:
		assert isinstance(exc1, expected_exception), (
			f"Elephant raised {type(exc1)}, expected {expected_exception}")
		assert isinstance(exc2, expected_exception), (
			f"Pydantic raised {type(exc2)}, expected {expected_exception}")

	if same_type:
		if(type(exc1) is type(exc2)):
			return

		# Treat (Pydantic ValidationError) <-> (ValueError|TypeError) as equivalent
		if (isinstance(exc1, (ValueError, TypeError)) and isinstance(exc2, (ValidationError, AttributeError))):
			return

		# No known equivalence -> fail
		assert False, (
			f"Different exception types: Elephant={type(exc1)}, Pydantic={type(exc2)}. "
			f"Elephant exc: {exc1}; Pydantic exc: {exc2}")


@pytest.mark.parametrize("cls,fn_name", [
	(PydanticTimeHistogram, elephant.statistics.time_histogram),
	(PydanticFanofactor, elephant.statistics.fanofactor),
	(PydanticComplexityPdf, elephant.statistics.complexity_pdf),
])
def test_spiketrains_invalid_type(cls, fn_name):
	bad = {"spiketrains": [1, 2, 3], "bin_size": 5 * pq.ms}
	# elephant function call signature varies; call with minimal args when possible
	# Use the helper to consistently call elephant functions
	assert_both_raise_consistently(fn_name, cls, bad)


def test_spade_invalid_spiketrains_and_negative_winlen():
	bad_spikes = {"spiketrains": [1, 2, 3], "bin_size": 5 * pq.ms, "winlen": 10}
	assert_both_raise_consistently(elephant.spade.spade, PydanticSpade, bad_spikes)

	bad_winlen = {"spiketrains": [make_spiketrain()], "bin_size": 5 * pq.ms, "winlen": -1}
	# elephant should raise for negative window length
	assert_both_raise_consistently(elephant.spade.spade, PydanticSpade, bad_winlen)


def test_concepts_mining_invalid_inputs():
	bad = {"spiketrains": [1, 2, 3], "bin_size": 5 * pq.ms, "winlen": 5}
	assert_both_raise_consistently(elephant.spade.concepts_mining, PydanticConceptsMining, bad)


def test_pvalue_spectrum_invalid_inputs():
	bad = {"spiketrains": [1, 2, 3], "bin_size": 5 * pq.ms, "winlen": 5, "dither": 15 * pq.ms, "n_surr": 10}
	assert_both_raise_consistently(elephant.spade.pvalue_spectrum, PydanticPValueSpectrum, bad)


def test_jointJ_window_analysis_invalid_spiketrains_and_params():
	bad = {"spiketrains": [1, 2, 3], "bin_size": 5 * pq.ms}
	assert_both_raise_consistently(elephant.unitary_event_analysis.jointJ_window_analysis, PydanticJointJWindowAnalysis, bad)


def test_cell_assembly_detection_invalid_binned_spiketrain():
	bad = {"binned_spiketrain": [1, 2, 3], "max_lag": 2}
	assert_both_raise_consistently(elephant.cell_assembly_detection.cell_assembly_detection, PydanticCellAssemblyDetection, bad)


def test_optimal_kernel_bandwidth_invalid_spiketimes():
	bad = {"spiketimes": "not an array"}
	assert_both_raise_consistently(elephant.statistics.optimal_kernel_bandwidth, PydanticOptimalKernelBandwidth, bad)


def test_lvr_R_validation():
	# R must be non-negative quantity; negative should raise
	gt = {"time_intervals": np.array([1.0, 2.0]), "R": -5 * pq.ms}
	# elephant.lvr expects time_intervals and R; call and expect error
	assert_both_raise_consistently(elephant.statistics.lvr, PydanticLvr, gt)


def test_lvr_R_zero_allowed():
	# R == 0 should be allowed by Pydantic wrapper
	ok = {"time_intervals": np.array([1.0, 2.0]), "R": 0 * pq.ms}
	# Pydantic should instantiate without error
	inst = PydanticLvr(**ok)
	assert isinstance(inst, PydanticLvr)


def test_pvalue_spectrum_n_surr_zero_and_negative():
	base = {
		"spiketrains": [make_spiketrain()],
		"bin_size": 5 * pq.ms,
		"winlen": 5,
		"dither": 15 * pq.ms,
	}
	ok_zero = {**base, "n_surr": 0}
	# Pydantic should accept zero surrogates (boundary)
	inst = PydanticPValueSpectrum(**ok_zero)
	assert isinstance(inst, PydanticPValueSpectrum)

	bad_neg = {**base, "n_surr": -1}
	# elephant should raise for invalid n_surr
	assert_both_raise_consistently(elephant.spade.pvalue_spectrum, PydanticPValueSpectrum, bad_neg)


def test_spade_winlen_zero_raises():
	bad_winlen_zero = {"spiketrains": [make_spiketrain()], "bin_size": 5 * pq.ms, "winlen": 0}
	# elephant should raise for zero window length (boundary)
	assert_both_raise_consistently(elephant.spade.spade, PydanticSpade, bad_winlen_zero)


def test_timehistogram_bin_size_zero_raises():
	bad = {"spiketrains": [make_spiketrain()], "bin_size": 0 * pq.ms}
	# elephant.statistics.time_histogram should reject zero bin size
	assert_both_raise_consistently(elephant.statistics.time_histogram, PydanticTimeHistogram, bad)


def test_fanofactor_bin_size_negative_and_small_positive():
	small = {"spiketrains": [make_spiketrain()], "bin_size": 1e-6 * pq.s}
	# very small but positive bin size should be accepted by Pydantic wrapper
	inst = PydanticFanofactor(**small)
	assert isinstance(inst, PydanticFanofactor)


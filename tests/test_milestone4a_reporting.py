"""Reporting uses actual validity masks and checked numerical evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.milestone4a import summarize_candidate, validation_summary, write_report


def payload():
    return {
        'configuration': {'seed': 7},
        'candidate_results': {'zero': {
            'candidate_valid': True, 'status': 'valid', 'requested_observations': 1,
            'valid_observations': 1, 'masked_observations': 0, 'rejected_observations': 0,
            'innovation_quadratic': 2., 'logdet_diagnostic': -1., 'gaussian_diagnostic': 1.,
            'numpy_consistency': {'passed': True}}},
        'basis_constant_reconstruction_max_abs_error': 1e-16,
        'gradient_validation': {'results': {'nonconstant': {'1e-05': {
            'jax_gradient': [1., -2.], 'central_difference_gradient': [1., -2.],
            'all_candidates_valid': True, 'max_absolute_error': 0., 'max_relative_error': 0.}}}},
        'test_execution': {'executed_by_benchmark': False,
                           'historical_record': 'Previously recorded: 7 focused / 205 total tests passed.',
                           'current_run_record': None}}


@pytest.mark.parametrize('failure', ['candidate', 'gradient', 'nonfinite_gradient', 'perturbation', 'numpy'])
def test_report_cannot_claim_success_for_failed_validation(tmp_path, failure):
    record = deepcopy(payload())
    gradient = record['gradient_validation']['results']['nonconstant']['1e-05']
    if failure == 'candidate':
        record['candidate_results']['zero']['candidate_valid'] = False
        record['candidate_results']['zero']['status'] = 'INVALID'
    elif failure == 'gradient':
        gradient['central_difference_gradient'][0] += 1.
        gradient['max_absolute_error'] = 1.
    elif failure == 'nonfinite_gradient':
        gradient['jax_gradient'][0] = np.nan
    elif failure == 'perturbation':
        gradient['all_candidates_valid'] = False
    else:
        record['candidate_results']['zero']['numpy_consistency']['passed'] = False
    path = tmp_path/'report.md'
    write_report(path, record)
    report = path.read_text()
    assert 'Numerical benchmark validation: **FAILED**' in report
    assert not all(validation_summary(record).values())
    assert 'Current run test record supplied separately:' in report
    assert 'does not execute pytest' in report
    assert '205 total tests passed' in report  # Explicit historical provenance.
    assert 'null' in report  # No current tests are fabricated.


def test_valid_report_has_derived_numerical_verdict(tmp_path):
    record = payload()
    path = tmp_path/'report.md'
    write_report(path, record)
    assert all(validation_summary(record).values())
    assert 'Numerical benchmark validation: **PASSED**' in path.read_text()


def test_statistics_use_valid_mask_and_suppress_invalid_candidate():
    # Slots at one timestamp: valid, geometry-rejected, masked. Placeholder
    # values are deliberately huge to expose an accidental requested-mask sum.
    result = SimpleNamespace(
        candidate_valid=False, requested_count=2, valid_count=1, masked_count=1,
        rejected_count=1, invalid_geometry_count=1, numerical_rejection_count=0,
        states_finite=True, covariances_symmetric=True, covariances_psd=True,
        positive_definite=True, valid_mask=np.array([[True, False, False]]),
        innovations=np.array([[[.2, .1], [999., 999.], [999., 999.]]]),
        innovation_covariances=np.tile(np.eye(2), (1, 3, 1, 1)),
        final_pose=np.zeros(3), final_covariance=np.eye(3),
        innovation_quadratic=.05, logdet_diagnostic=0., gaussian_diagnostic=.05)
    summary = summarize_candidate(result, np.zeros((1, 3)))
    assert summary['innovation_statistics'] is None
    assert 'innovation_quadratic' not in summary
    assert summary['requested_observations'] == 2 and summary['rejected_observations'] == 1
    # A valid evaluation with two masked slots excludes both placeholders.
    result.candidate_valid = True
    result.requested_count = 1
    result.masked_count = 2
    result.rejected_count = result.invalid_geometry_count = 0
    summary = summarize_candidate(result, np.zeros((1, 3)))
    assert summary['innovation_statistics']['count'] == 1
    np.testing.assert_array_equal(summary['innovation_statistics']['mean_range_bearing'], [.2, .1])
    assert summary['numpy_consistency']['passed']

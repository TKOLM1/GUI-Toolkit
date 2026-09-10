"""Unit tests for the thread-budget arbiter (ml.parallel) that prevents oversubscription."""

from __future__ import annotations

from ml.parallel import inner_fit_limit, inner_fit_n_jobs


def test_default_is_unlimited():
    # No stage fanning out: a self-parallelising estimator keeps its own n_jobs (the -1 default).
    assert inner_fit_n_jobs(-1) == -1
    assert inner_fit_n_jobs(8) == 8


def test_limit_caps_inner_n_jobs_within_block():
    # Inside the context the cap overrides the factory's default, so the forest builds single-threaded.
    with inner_fit_limit(1):
        assert inner_fit_n_jobs(-1) == 1
        assert inner_fit_n_jobs(8) == 1
    # Restored to unlimited on exit.
    assert inner_fit_n_jobs(-1) == -1


def test_limit_restores_previous_on_exception():
    try:
        with inner_fit_limit(1):
            assert inner_fit_n_jobs(-1) == 1
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert inner_fit_n_jobs(-1) == -1  # cleaned up even on error


def test_nesting_restores_outer_value_not_unlimited():
    # An inner limit must restore the *outer* limit on exit, not blow it away to unlimited.
    with inner_fit_limit(2):
        assert inner_fit_n_jobs(-1) == 2
        with inner_fit_limit(1):
            assert inner_fit_n_jobs(-1) == 1
        assert inner_fit_n_jobs(-1) == 2  # outer limit survives the inner block
    assert inner_fit_n_jobs(-1) == -1

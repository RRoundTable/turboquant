"""Phase 15a dispatcher tests: verify runtime arch-based kernel selection.

Does NOT require the v5/v6/v7 kernel to be loaded — tests the dispatch
*logic*, including the force-override path and the missing-op error.
"""
import os
from unittest import mock

import pytest
import torch


@pytest.fixture(autouse=True)
def _clear_cache():
    """`pick_decode_op` is lru_cached; clear between tests so env-var changes
    take effect."""
    from turboquant.decoder_dispatch import pick_decode_op, _device_capability
    pick_decode_op.cache_clear()
    _device_capability.cache_clear()
    yield
    pick_decode_op.cache_clear()
    _device_capability.cache_clear()


def test_dispatch_picks_v5_on_sm80(monkeypatch):
    """On SM80 with the v5 op registered, pick_decode_op returns v5."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = torch.cuda.get_device_capability()
    if cc < (8, 0):
        pytest.skip(f"requires SM80+, got {cc}")

    # Force the v5 op to be considered "registered" for the dispatcher.
    # (In a real test we'd import turboquant.decode_kernel_v4 first to load
    # the module, but for the logic test we just stub _has_op.)
    from turboquant import decoder_dispatch as dd
    monkeypatch.setattr(dd, "_has_op",
        lambda lib, name: lib == "turboquant_v5" and name == "decode_v5_from_cache_paged_splitkv_ws")
    # Also stub _v5_op so we don't hit the unregistered real attr.
    sentinel_op = object()
    monkeypatch.setattr(dd, "_v5_op", lambda: sentinel_op)
    dd.pick_decode_op.cache_clear()

    op, name = dd.pick_decode_op()
    assert name == "v5_paged_split_ampere"
    assert op is sentinel_op


def test_dispatch_falls_back_to_v4_below_sm80(monkeypatch):
    """On pre-SM80 hardware, dispatch returns the v4 fallback."""
    from turboquant import decoder_dispatch as dd
    monkeypatch.setattr(dd, "_device_capability", lambda: (7, 5))
    monkeypatch.setattr(dd, "_has_op", lambda *_a: False)
    dd.pick_decode_op.cache_clear()

    op, name = dd.pick_decode_op()
    assert name == "v4_fallback_generic"


def test_dispatch_picks_v6_on_sm90_when_available(monkeypatch):
    from turboquant import decoder_dispatch as dd
    monkeypatch.setattr(dd, "_device_capability", lambda: (9, 0))
    monkeypatch.setattr(dd, "_has_op",
        lambda lib, name: lib == "turboquant_v6")
    sentinel = object()
    monkeypatch.setattr(dd, "_v6_op", lambda: sentinel)
    dd.pick_decode_op.cache_clear()

    op, name = dd.pick_decode_op()
    assert name == "v6_wgmma_hopper"
    assert op is sentinel


def test_dispatch_picks_v7_on_sm100_when_available(monkeypatch):
    from turboquant import decoder_dispatch as dd
    monkeypatch.setattr(dd, "_device_capability", lambda: (10, 0))
    monkeypatch.setattr(dd, "_has_op",
        lambda lib, name: lib == "turboquant_v7")
    sentinel = object()
    monkeypatch.setattr(dd, "_v7_op", lambda: sentinel)
    dd.pick_decode_op.cache_clear()

    op, name = dd.pick_decode_op()
    assert name == "v7_fp4_blackwell"
    assert op is sentinel


def test_dispatch_degrades_to_v5_on_sm90_without_v6(monkeypatch):
    """SM90 hardware but v6 kernel not compiled: must fall through to v5."""
    from turboquant import decoder_dispatch as dd
    monkeypatch.setattr(dd, "_device_capability", lambda: (9, 0))
    # v6 not registered, v5 is
    monkeypatch.setattr(dd, "_has_op",
        lambda lib, name: lib == "turboquant_v5")
    sentinel = object()
    monkeypatch.setattr(dd, "_v5_op", lambda: sentinel)
    dd.pick_decode_op.cache_clear()

    op, name = dd.pick_decode_op()
    assert name == "v5_paged_split_ampere"


def test_force_kernel_env_override(monkeypatch):
    """TQ_FORCE_KERNEL=v5 picks v5 regardless of detected arch."""
    from turboquant import decoder_dispatch as dd
    monkeypatch.setattr(dd, "_device_capability", lambda: (10, 0))  # pretend Blackwell
    monkeypatch.setenv("TQ_FORCE_KERNEL", "v5")
    sentinel = object()
    monkeypatch.setattr(dd, "_v5_op", lambda: sentinel)
    dd.pick_decode_op.cache_clear()

    op, name = dd.pick_decode_op()
    assert name == "v5_paged_split_ampere"
    assert op is sentinel


def test_force_kernel_unknown_raises(monkeypatch):
    from turboquant import decoder_dispatch as dd
    monkeypatch.setenv("TQ_FORCE_KERNEL", "v99")
    dd.pick_decode_op.cache_clear()
    with pytest.raises(ValueError, match="not recognised"):
        dd.pick_decode_op()


def test_force_kernel_unavailable_raises(monkeypatch):
    """TQ_FORCE_KERNEL=v6 on a machine where v6 isn't compiled should give
    a clear error, not a bare AttributeError."""
    from turboquant import decoder_dispatch as dd
    monkeypatch.setenv("TQ_FORCE_KERNEL", "v6")
    def _raise_unreg():
        raise AttributeError("no such op")
    monkeypatch.setattr(dd, "_v6_op", _raise_unreg)
    dd.pick_decode_op.cache_clear()
    with pytest.raises(RuntimeError, match="not registered"):
        dd.pick_decode_op()


def test_cache_hit_returns_same_tuple():
    """Second call hits lru_cache, returns the same object."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from turboquant.decoder_dispatch import pick_decode_op
    # With v5 module not imported, this falls back; still must be deterministic.
    a = pick_decode_op()
    b = pick_decode_op()
    assert a == b

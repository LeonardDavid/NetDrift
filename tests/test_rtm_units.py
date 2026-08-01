import numpy as np
import pytest
import torch

from netdrift.faults.base import FaultCtx
from netdrift.faults.rtm_misalignment import RTMConfig, RTMMisalignmentFault


def _ctx():
    return FaultCtx(layer_id=1, layer_name="l1", nr_run=1, training=False, bits=1,
                    extra={"rt_mapping": "UNITS", "kernel_mapping": None,
                           "kernel_size": None, "base_layout": "ROW"})


def test_init_state_requires_base_layout():
    f = RTMMisalignmentFault(RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True))
    ctx = _ctx()
    ctx.extra["base_layout"] = None
    with pytest.raises(ValueError, match="base_layout"):
        f.init_state((4, 64), ctx)


def test_init_state_returns_units_mapping():
    f = RTMMisalignmentFault(RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True))
    st = f.init_state((4, 64), _ctx())
    assert st.rt_mapping == "UNITS"
    assert st.base_mapping == "ROW"
    assert st.unit_buckets is None      # built lazily on first inject


def test_units_rejects_per_forward_weight_encoder():
    class _StubEncoder:  # only identity matters; never invoked
        pass
    with pytest.raises(ValueError, match="per_forward"):
        RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True,
                  weight_encoder=_StubEncoder(), weight_encoder_mode="per_forward")


def test_units_allows_once_mode_encoder_at_config_level():
    # 'once' is not rejected here; the runner's own guard handles policy.
    class _StubEncoder:
        pass
    RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True,
              weight_encoder=_StubEncoder(), weight_encoder_mode="once")


@pytest.mark.cuda
def test_units_bit_exact_at_zero_error():
    torch.manual_seed(0)
    w = torch.where(torch.rand(8, 64) < 0.5, 1.0, -1.0)
    for threshold, mp, pg in [(1, 1, 0), (2, 2, 1), (2, 1, 0), (4, 1, 0)]:
        cfg = RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True,
                        units_threshold=threshold, units_max_period=mp,
                        units_pool_guard=pg)
        f = RTMMisalignmentFault(cfg)
        ctx = _ctx()
        st = f.init_state(tuple(w.shape), ctx)
        new_w, st2, _stats = f.inject(w, st, ctx)
        assert torch.equal(new_w, w), f"not bit-exact at threshold={threshold}"
        assert st2.unit_buckets is not None


@pytest.mark.cuda
def test_units_structure_cached_across_calls():
    torch.manual_seed(1)
    w = torch.where(torch.rand(4, 64) < 0.5, 1.0, -1.0)
    cfg = RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True,
                    units_threshold=2, units_max_period=2, units_pool_guard=1)
    f = RTMMisalignmentFault(cfg)
    ctx = _ctx()
    st = f.init_state(tuple(w.shape), ctx)
    _w1, st1, _ = f.inject(w, st, ctx)
    _w2, st2, _ = f.inject(w, st1, ctx)
    assert st2.unit_buckets is st1.unit_buckets


@pytest.mark.cuda
def test_units_threshold1_matches_block_immunity_under_saturate():
    """At threshold=1 units is structurally identical to BLOCK, and BLOCK under
    saturate is fault-immune: every wire is one same-sign run plus a same-sign
    guard band, so a clamped read always returns the correct sign. Both arms
    must therefore flip exactly zero bits -- that shared zero IS the property
    being checked, not a vacuous result.
    """
    torch.manual_seed(3)
    w = torch.where(torch.rand(8, 64) < 0.5, 1.0, -1.0)
    common = dict(rt_size=64, rt_error=1e-3, edge_mode="saturate")
    fb = RTMMisalignmentFault(RTMConfig(block_mapping=True, **common))
    fu = RTMMisalignmentFault(RTMConfig(units_mapping=True, units_threshold=1,
                                        units_max_period=1, units_pool_guard=0,
                                        **common))
    cb, cu = _ctx(), _ctx()
    cb.extra["rt_mapping"] = "BLOCK"
    wb, _, _ = fb.inject(w, fb.init_state(tuple(w.shape), cb), cb)
    wu, _, _ = fu.inject(w, fu.init_state(tuple(w.shape), cu), cu)
    assert int((wb != w).sum()) == 0, "BLOCK under saturate must be fault-immune"
    assert int((wu != w).sum()) == 0, "units@threshold=1 must match BLOCK's immunity"


@pytest.mark.cuda
def test_units_threshold1_matches_block_under_random_edge_mode():
    """edge_mode='random' is where BLOCK is NOT immune -- an out-of-bounds read
    returns a random value rather than saturating -- so this is the only regime
    where a flip-count comparison is meaningful. The two paths do not share an
    RNG stream ordering, so compare magnitudes, not exact equality.
    """
    torch.manual_seed(3)
    w = torch.where(torch.rand(32, 64) < 0.5, 1.0, -1.0)
    # BLOCK only faults at all under edge_mode="random"; each wire's drift
    # probability is 1-(1-rt_error)**P where P is that wire's own physical
    # length, and at threshold=1 most wires are only 1-4 cells long (mean run
    # length ~2), so rt_error must be high or the assert nb > 0 guard below is
    # flaky rather than the implementation being wrong (measured ~5% failure
    # rate on correct code at rt_error=1e-2 on an 8x64 input).
    common = dict(rt_size=64, rt_error=0.1, edge_mode="random")
    fb = RTMMisalignmentFault(RTMConfig(block_mapping=True, **common))
    fu = RTMMisalignmentFault(RTMConfig(units_mapping=True, units_threshold=1,
                                        units_max_period=1, units_pool_guard=0,
                                        **common))
    cb, cu = _ctx(), _ctx()
    cb.extra["rt_mapping"] = "BLOCK"
    wb, _, _ = fb.inject(w, fb.init_state(tuple(w.shape), cb), cb)
    wu, _, _ = fu.inject(w, fu.init_state(tuple(w.shape), cu), cu)
    nb, nu = int((wb != w).sum()), int((wu != w).sum())
    assert nb > 0, "random edge mode should fault BLOCK; raise rt_error if not"
    assert nu > 0, "random edge mode should fault units@threshold=1"
    assert abs(nb - nu) <= max(8, int(0.5 * nb)), (nb, nu)


@pytest.mark.cuda
def test_units_period1_wires_stay_immune_under_faults():
    """Under saturate, an isolated period-1 wire is fault-immune by construction:
    every cell on it shares a sign and the read clamps inside the wire. Only
    cells living on pooled period-2 wires may flip. This is the first test to
    exercise threshold=2 + max_period=2 + pool_guard=1 at rt_error > 0.
    """
    from netdrift.faults.packing import build_unit_wires

    torch.manual_seed(11)
    # Values must be exactly +/-1 so "sign preserved" == "value preserved";
    # a per-channel scale would let a same-sign shift change the magnitude.
    w = torch.where(torch.rand(32, 64) < 0.5, 1.0, -1.0)

    wires = build_unit_wires(w, 64, threshold=2, max_period=2, pool_guard=1)
    isolated = {(r, c) for x in wires if x.period == 1
                for r, c in zip(x.rows, x.cols) if r >= 0}
    pooled = {(r, c) for x in wires if x.period == 2
              for r, c in zip(x.rows, x.cols) if r >= 0}
    assert isolated and pooled, "config did not produce both wire kinds"

    # rt_error is set high (not the usual 1e-2) because this config yields only
    # ~12 pooled period-2 wires at this size, and the kernel's RNG is seeded
    # with random.randint() (rtm_misalignment.py, _run_rtm_kernels) rather than
    # torch.manual_seed -- it is genuinely non-deterministic per run. At 1e-2
    # the chance that none of the ~12 wires drifts is ~4e-4 (worse in practice,
    # since the offset random-walks and can return to 0 within the same call),
    # which would fail the vacuity assertion below on a correct implementation.
    # 5e-2 drops that to ~8e-18.
    cfg = RTMConfig(rt_size=64, rt_error=5e-2, units_mapping=True,
                    units_threshold=2, units_max_period=2, units_pool_guard=1,
                    edge_mode="saturate")
    f = RTMMisalignmentFault(cfg)
    ctx = _ctx()
    st = f.init_state(tuple(w.shape), ctx)
    new_w, _st2, _stats = f.inject(w, st, ctx)

    # base_layout=ROW on a 2D weight is an identity reshape, so positions map 1:1.
    flipped = {(int(r), int(c)) for r, c in zip(*torch.nonzero(new_w != w, as_tuple=True))}
    assert flipped, "no faults injected -- raise rt_error or the test proves nothing"
    assert not (flipped & isolated), sorted(flipped & isolated)[:8]
    assert flipped <= pooled


def _conv_ctx():
    return FaultCtx(
        layer_id=1, layer_name="c", nr_run=1, training=False,
        extra={"rt_mapping": "UNITS", "kernel_mapping": "ROW",
               "kernel_size": 3, "base_layout": "ROW"},
    )


@pytest.mark.cuda
def test_units_conv_weight_roundtrips_at_zero_error():
    """4D conv weights go through kernel-mapping + UNITS and are bit-exact at
    rt_error=0, mirroring
    tests/test_rtm_block_gpu.py::test_block_conv_weight_roundtrips_at_zero_error.

    Every other units test in this file uses a 2D weight (8x64, 4x64, 32x64),
    but 6 of the 7 unprotected VGG7 layers are convolutions, and the conv
    branch of ``_layout_weight_for_racetrack``/``undo`` (``_rearrange_kernel``
    -> ``reshape(out_c, -1)`` -> parse -> scatter -> ``_restore_kernel``) is
    exactly the surface the accepted BLOCK/UNITS duplication risks silently
    diverging on -- BLOCK has a dedicated conv test and units did not.
    """
    torch.manual_seed(3)
    w = torch.sign(torch.randn(4, 4, 3, 3))
    w[w == 0] = -1.0
    cfg = RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True,
                    units_threshold=2, units_max_period=2, units_pool_guard=1,
                    track_bitflips=True)
    f = RTMMisalignmentFault(cfg)
    ctx = _conv_ctx()
    st = f.init_state(tuple(w.shape), ctx)
    new_w, st2, stats = f.inject(w, st, ctx)
    assert new_w.shape == w.shape
    assert torch.equal(new_w, w)
    assert stats.bitflips == 0
    assert st2.unit_buckets is not None


@pytest.mark.cuda
def test_units_conv_weight_faults_confined_to_pooled_cells():
    """Under a nonzero fault rate, the conv path's 4D shape still round-trips
    and flips stay confined to pooled (period-2) cells, exactly as
    ``test_units_period1_wires_stay_immune_under_faults`` shows for a plain 2D
    weight -- but exercised through the kernel-mapping reshape/restore
    bracket that test never touches.

    P(zero events), derived correctly (see the fix report for the full
    derivation and the mistake this corrects): the naive formula used
    elsewhere in this project, "a wire drifts w.p. 1-(1-rt_error)**P", is only
    a LOWER BOUND on the true vacuous-pass probability, not the value itself
    -- it counts only the case where zero fault events fire during the wire's
    P reads, but ``calc_index_offset_kernel``'s saturate-mode offset is a
    BOUNDED random walk (clamped to roughly [-P/2, P/2]), so (a) the walk can
    fire several fault events and still random-walk back to offset 0, and (b)
    even at nonzero offset, spec 3.1's clamped-prefix wrong-slot set can land
    entirely on filler/guard cells, which are discarded on scatter-back and
    never register as a weight flip. A Monte-Carlo replay of the exact
    walk-then-read model (the project's own kernel-validated ``_read`` mirror
    from ``tests/test_units_guard_band.py``, driven by the same
    fire-then-direction Markov chain as ``calc_index_offset_kernel``'s
    ``edge_mode=1`` branch, ordinary Python RNG -- not bit-exact vs xoroshiro128p,
    but the same process, so adequate for choosing a safe rate) measured a
    PER-WIRE vacuous rate of ~9-33% for a single P=64 guarded pooled wire at
    rt_error in [0.1, 0.5] -- it does not fall toward zero as rt_error rises,
    because a higher rate also raises the chance the walk returns near 0 or
    lands on an all-guard prefix. The rate clusters at two values rather than
    one: spec 3.1 says an ODD offset makes the whole unclamped interior wrong
    (a real-cell flip is then all but certain, giving the low ~9% cluster --
    that residual is walks landing back at exactly off=0), while an EVEN
    offset is only wrong in the short clamped prefix, which occasionally
    lands entirely on filler/guard cells (the higher ~23-33% cluster). A
    single pooled wire therefore CANNOT reach a ~1e-6 floor at any rt_error;
    using (4,4,3,3) (one pooled wire) would have silently reproduced this
    branch's earlier 5.4%/0.04% vacuous-pass incidents. The fix is more
    independent pooled wires, whose per-wire vacuous events must then all
    coincide: (16,16,3,3) yields 12-14 independent P=64 guarded pooled wires
    at this config depending on the sign pattern (checked across 12 seeds via
    the same importlib-bypass harness -- never below 12, so this is a
    structural property of the (16,144) base-2D view at rt_size=64, not a
    lucky draw for seed=3), and multiplying the per-wire rates measured for
    seed=3's 14 wires at rt_error=0.3 gives an ensemble P(zero flips) of
    ~1.6e-12 -- comfortably below the ~1e-6 floor this task asked for. A
    deliberately pessimistic cross-check that stacks two independent
    worst-cases -- the worst single-wire rate observed anywhere (~0.33)
    raised to the WORST seed's wire count (12), i.e. assuming every wire
    simultaneously hits the highest rate ever measured -- gives
    0.33**12 ~= 3.2e-6. That bound sits slightly ABOVE the 1e-6 floor, but it
    is not the expected rate; the measured 1.6e-12 (real per-wire rates, not
    a uniform worst-case) is. Both numbers are far below anything this
    project has previously shipped (5.4%, 0.04%).
    """
    from netdrift.faults.layout import _layout_weight_for_racetrack
    from netdrift.faults.packing import build_unit_wires

    torch.manual_seed(3)
    w = torch.sign(torch.randn(16, 16, 3, 3))
    w[w == 0] = -1.0

    w_2d, _undo = _layout_weight_for_racetrack(w, rt_mapping="ROW", kernel_mapping="ROW")
    wires = build_unit_wires(w_2d, 64, threshold=2, max_period=2, pool_guard=1)
    isolated = {(r, c) for x in wires if x.period == 1
                for r, c in zip(x.rows, x.cols) if r >= 0}
    pooled = {(r, c) for x in wires if x.period == 2
              for r, c in zip(x.rows, x.cols) if r >= 0}
    assert isolated and pooled, "config did not produce both wire kinds"

    cfg = RTMConfig(rt_size=64, rt_error=0.3, units_mapping=True,
                    units_threshold=2, units_max_period=2, units_pool_guard=1,
                    edge_mode="saturate")
    f = RTMMisalignmentFault(cfg)
    ctx = _conv_ctx()
    st = f.init_state(tuple(w.shape), ctx)
    new_w, _st2, _stats = f.inject(w, st, ctx)
    assert new_w.shape == w.shape

    # Re-derive the post-fault 2D view through the SAME helper rather than
    # assuming reshape is a bijection by inspection -- robust to km/rt_mapping
    # combinations that aren't a plain reshape.
    new_w_2d, _undo2 = _layout_weight_for_racetrack(new_w, rt_mapping="ROW", kernel_mapping="ROW")
    flipped = {(int(r), int(c)) for r, c in zip(*torch.nonzero(new_w_2d != w_2d, as_tuple=True))}
    assert flipped, "no faults injected -- raise rt_error or the test proves nothing"
    assert not (flipped & isolated), sorted(flipped & isolated)[:8]
    assert flipped <= pooled

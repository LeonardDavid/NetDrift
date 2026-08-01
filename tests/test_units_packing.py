import numpy as np
import torch

from netdrift.faults.packing import build_unit_buckets, build_unit_wires


def _sign(v):
    return 1 if v > 0 else -1


def test_isolated_wire_has_period1_guard_band():
    # One len-3 run of +1, rt_size 4, threshold 1 -> P=4, pad cell holds +1.
    w = torch.tensor([[1.0, 1.0, 1.0, -1.0]])
    wires = build_unit_wires(w, 4, threshold=1, max_period=1, pool_guard=0)
    three = [x for x in wires if sum(r >= 0 for r in x.rows) == 3][0]
    assert three.period == 1
    assert len(three.rows) == 4                      # next_pow2(3)
    assert three.rows[3] == -1 and three.cols[3] == -1
    assert three.fill[3] == 1.0                      # own sign, not zero


def test_length1_isolated_wire_has_no_pad():
    # next_pow2(1) == 1, so a len-1 run gets a 1-cell wire with no guard band.
    w = torch.tensor([[1.0, -1.0]])
    wires = build_unit_wires(w, 2, threshold=1, max_period=1, pool_guard=0)
    assert all(len(x.rows) == 1 for x in wires)
    assert all(x.rows[0] >= 0 for x in wires)


def test_pooled_wire_is_period2_with_guard():
    # runs [1,1,3,1,1]: threshold=2 isolates the len-3 run and pools the rest
    # into two fragments. With pool_guard=1 the assembled pooled wire must
    # alternate at EVERY slot, including across the fragment junction.
    w = torch.tensor([[1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]])
    wires = build_unit_wires(w, 7, threshold=2, max_period=2, pool_guard=1)
    pooled = [x for x in wires if x.period == 2]
    assert len(pooled) == 1
    p = pooled[0]
    vals = []
    for j in range(len(p.rows)):
        vals.append(p.fill[j] if p.rows[j] < 0 else float(w[p.rows[j], p.cols[j]]))
    s = [_sign(v) for v in vals]
    assert all(s[i] != s[i + 1] for i in range(len(s) - 1)), s


def test_pooled_wire_without_guard_breaks_phase():
    # The spec's argument: a fragment ends with the sign opposite the isolated
    # block that follows it, and the next fragment begins with that same sign,
    # so an unguarded junction is always a phase break.
    w = torch.tensor([[1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]])
    wires = build_unit_wires(w, 7, threshold=2, max_period=1, pool_guard=0)
    # the pooled wire is the one holding all 3 pooled cells (the isolated len-3
    # run is period-1 and lives on its own wire)
    p = [x for x in wires if x.period == 0][0]
    vals = [p.fill[j] if p.rows[j] < 0 else float(w[p.rows[j], p.cols[j]])
            for j in range(len(p.rows))]
    s = [_sign(v) for v in vals]
    assert any(s[i] == s[i + 1] for i in range(len(s) - 1)), s


def test_pooled_wire_capped_at_rt_size():
    # 64 alternating cells at threshold=2 -> pooled wires never exceed rt_size.
    w = torch.tensor([[1.0, -1.0] * 64])
    wires = build_unit_wires(w, 64, threshold=2, max_period=2, pool_guard=1)
    assert all(len(x.rows) <= 64 for x in wires)


def test_every_real_cell_appears_exactly_once():
    torch.manual_seed(0)
    w = torch.where(torch.rand(7, 37) < 0.5, 1.0, -1.0)
    for threshold in (1, 2, 3, 4, 8):
        mp, pg = (2, 1) if threshold == 2 else (1, 0)
        wires = build_unit_wires(w, 8, threshold=threshold,
                                 max_period=mp, pool_guard=pg)
        seen = []
        for x in wires:
            for r, c in zip(x.rows, x.cols):
                if r >= 0:
                    seen.append((r, c))
        assert len(seen) == len(set(seen)), f"duplicate cell at threshold={threshold}"
        assert set(seen) == {(r, c) for r in range(7) for c in range(37)}


def test_period2_invariant_holds_at_scale():
    """The invariant that a small hand-built case cannot check.

    A fixed one-guard-per-junction rule passes the 7-cell case above but breaks
    ~34% of junctions on real data, because whether a junction needs a guard
    depends on the parity of how many isolated runs separate the two fragments.
    This test is what catches that; do not delete it.
    """
    torch.manual_seed(0)
    w = torch.where(torch.rand(128, 4608) < 0.5, 1.0, -1.0)
    signs = torch.where(w > 0, 1, -1).tolist()
    wires = build_unit_wires(w, 64, threshold=2, max_period=2, pool_guard=1)
    pooled = [x for x in wires if x.period == 2]
    assert pooled, "expected pooled period-2 wires"
    for x in pooled:
        vals = [x.fill[j] if x.rows[j] < 0 else signs[x.rows[j]][x.cols[j]]
                for j in range(len(x.rows))]
        s = [1 if v > 0 else -1 for v in vals]
        breaks = [i for i in range(len(s) - 1) if s[i] == s[i + 1]]
        assert not breaks, f"phase break at slots {breaks[:5]}"


def test_guard_rate_matches_the_closed_form():
    # P(guard needed) = 1 / (1 + P(a run is isolated)); ~0.67 for random signs.
    from netdrift.faults.units import parse_units
    torch.manual_seed(0)
    w = torch.where(torch.rand(64, 4608) < 0.5, 1.0, -1.0)
    iso, frags = parse_units(w, 64, threshold=2)
    n_runs = len(iso) + sum(len(f.cells) for f in frags)  # pooled runs are len 1
    p_iso = len(iso) / n_runs
    need = sum(1 for a, b in zip(frags[:-1], frags[1:]) if a.signs[-1] == b.signs[0])
    observed = need / (len(frags) - 1)
    assert abs(observed - 1.0 / (1.0 + p_iso)) < 0.02, (observed, p_iso)


def test_buckets_group_by_padded_length():
    torch.manual_seed(1)
    w = torch.where(torch.rand(4, 64) < 0.5, 1.0, -1.0)
    buckets = build_unit_buckets(w, 64, threshold=2, max_period=2, pool_guard=1)
    for p, b in buckets.items():
        assert b.weight_grid.shape[1] == p
        assert b.scatter_rows.shape == b.weight_grid.shape
        assert b.scatter_cols.shape == b.weight_grid.shape
        assert b.weight_grid.dtype == np.float32
        # real cells carry the weight value; fillers are masked out by -1
        mask = b.scatter_cols >= 0
        gr, gc = b.scatter_rows[mask], b.scatter_cols[mask]
        assert np.allclose(b.weight_grid[mask], w.numpy()[gr, gc])

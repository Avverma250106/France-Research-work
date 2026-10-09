"""Serving-parity test: does the live ring buffer reproduce offline features?

The simulation builds feature rows from a bounded 336-hour window. Training
builds them from each entity's entire history. Any feature that is not a pure
function of a bounded trailing window will differ between the two - and the
model will then be served inputs it was never trained on.

This test finds those features rather than assuming there are none.
"""

import numpy as np
import pandas as pd

import pipeline as P
from session_to_hourly import sessions_to_hourly, reindex_continuous
from sim.features_rt import FeatureBuffer, KEEP_HOURS
from sim import synth

TOL = 1e-9


def main():
    sessions = synth.generate(days=160, seed=3)
    sessions = sessions[sessions["entity_id"].isin(synth.PARIS_SITES)].reset_index(drop=True)
    hourly = reindex_continuous(sessions_to_hourly(sessions))

    offline = P.build_feature_table(hourly, sessions)
    offline = offline.sort_values(["entity_id", "hour"])

    # Verify the supervisor's leakage fix directly: the model feature at hour t
    # must equal the session count from t-SESSION_COUNT_LAG_HOURS, never the
    # contemporaneous count at t.
    raw = sessions.copy()
    raw["connect_time_utc"] = pd.to_datetime(raw["connect_time_utc"], utc=True)
    raw["hour"] = raw["connect_time_utc"].dt.floor("h")
    counts = (
        raw.groupby(["source_dataset", "entity_id", "hour"])
        .size()
        .rename("expected")
        .reset_index()
    )
    expected = hourly[["source_dataset", "entity_id", "hour"]].merge(
        counts, on=["source_dataset", "entity_id", "hour"], how="left"
    )
    expected["expected"] = expected["expected"].fillna(0)
    expected["expected"] = (
        expected.groupby(["source_dataset", "entity_id"])["expected"]
        .shift(P.SESSION_COUNT_LAG_HOURS)
    )
    check = offline[["source_dataset", "entity_id", "hour", P.SESSION_COUNT_FEATURE]].merge(
        expected, on=["source_dataset", "entity_id", "hour"], how="left"
    )
    valid = check[P.SESSION_COUNT_FEATURE].notna() & check["expected"].notna()
    if not valid.any():
        raise AssertionError("No rows available to verify historical session-count lag")
    max_diff = (
        check.loc[valid, P.SESSION_COUNT_FEATURE].astype(float)
        - check.loc[valid, "expected"].astype(float)
    ).abs().max()
    assert max_diff <= TOL, (
        f"{P.SESSION_COUNT_FEATURE} is not shifted by "
        f"{P.SESSION_COUNT_LAG_HOURS}h; max diff={max_diff}"
    )
    print(
        f"PASS  {P.SESSION_COUNT_FEATURE} is strictly historical "
        f"({P.SESSION_COUNT_LAG_HOURS}h lag)"
    )
    assert "sessions_started" not in offline.columns, (
        "Raw contemporaneous sessions_started must not be present"
    )
    last = offline.groupby("entity_id", as_index=False).tail(1).set_index("entity_id")

    prior = {str(e): [float(g["kwh_allocated"].sum()), int(len(g))]
             for e, g in hourly.groupby("entity_id")}
    buf = FeatureBuffer(synth.PARIS_SITES)
    buf.warm(hourly, sessions, prior=prior)
    live = buf.feature_rows().set_index("entity_id")

    cols = [c for c in offline.columns
            if c not in ("source_dataset", "entity_id", "site", "region", "hour")]

    print(f"Window: offline = full history ({len(hourly):,} rows), "
          f"live = last {KEEP_HOURS}h")
    print(f"Comparing {len(cols)} features across {len(live)} entities\n")

    mismatched = {}
    for c in cols:
        a = pd.to_numeric(last[c], errors="coerce").astype(float)
        b = pd.to_numeric(live.loc[a.index, c], errors="coerce").astype(float)
        both_nan = a.isna() & b.isna()
        diff = (a - b).abs()
        bad = (~both_nan) & ~(diff <= TOL)
        if bad.any():
            mismatched[c] = float(diff[bad].max())

    ok = [c for c in cols if c not in mismatched]
    print(f"PASS  {len(ok)} features identical to {TOL:g}")
    if mismatched:
        print(f"FAIL  {len(mismatched)} features differ:\n")
        for c, d in sorted(mismatched.items(), key=lambda kv: -kv[1]):
            print(f"   {c:24s} max abs diff {d:.6g}")
        print("\nThese are not pure functions of a bounded trailing window.")
        print("Serving them from a ring buffer feeds the model inputs that do")
        print("not match training.")
    return 0 if not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())

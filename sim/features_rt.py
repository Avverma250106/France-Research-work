"""Ring buffer producing live feature rows through the training code path.

Parity is by construction, not by reimplementation: the buffer rebuilds a small
hourly frame and calls pipeline.build_feature_table - the exact function
train_for_sim.py uses - then takes the last row per entity. Slower than a
hand-rolled incremental update, and correct, which matters more at 6 sites.
"""

from collections import deque

import numpy as np
import pandas as pd

import pipeline as P

KEEP_HOURS = 336  # covers lag_168h and the 168h rolling windows with headroom


class FeatureBuffer:
    def __init__(self, entities, source_dataset="paris", region="France"):
        self.entities = list(entities)
        self.source_dataset = source_dataset
        self.region = region
        self.hourly = {e: deque(maxlen=KEEP_HOURS) for e in self.entities}   # (hour, kwh)
        self.sessions = {e: deque(maxlen=4000) for e in self.entities}       # connect times
        # expanding_mean is not a function of a bounded window, so recomputing
        # it from the ring buffer would disagree with training. Carry it as
        # explicit state instead (test_parity.py asserts this).
        self.cum = {e: [0.0, 0] for e in self.entities}                      # [sum, count]

    def warm(self, hourly_df, sessions_df, prior=None):
        """Seed from real history so lag_168h is valid at t=0, not cold.

        `prior` is {entity: [sum, count]} over the FULL training history, which
        the ring buffer cannot see. Without it expanding_mean drifts.
        """
        if prior:
            for e in self.entities:
                if e in prior:
                    self.cum[e] = [float(prior[e][0]), int(prior[e][1])]
        for e in self.entities:
            h = hourly_df[hourly_df["entity_id"] == e].sort_values("hour")
            for _, r in h.tail(KEEP_HOURS).iterrows():
                self.hourly[e].append((pd.Timestamp(r["hour"]), float(r["kwh_allocated"])))
            s = sessions_df[sessions_df["entity_id"] == e]
            for t in s["connect_time_utc"]:
                self.sessions[e].append(pd.Timestamp(t))

    def last_hour(self):
        for e in self.entities:
            if self.hourly[e]:
                return self.hourly[e][-1][0]
        return None

    def append(self, entity, hour, kwh, connect_times):
        self.hourly[entity].append((pd.Timestamp(hour), float(kwh)))
        self.cum[entity][0] += float(kwh)
        self.cum[entity][1] += 1
        for t in connect_times:
            self.sessions[entity].append(pd.Timestamp(t))

    def _frames(self):
        rows = []
        for e in self.entities:
            for hour, kwh in self.hourly[e]:
                rows.append((self.source_dataset, e, e, self.region, hour, kwh))
        hourly = pd.DataFrame(rows, columns=[
            "source_dataset", "entity_id", "site", "region", "hour", "kwh_allocated"])

        srows = [(self.source_dataset, e, t)
                 for e in self.entities for t in self.sessions[e]]
        sessions = pd.DataFrame(srows, columns=["source_dataset", "entity_id", "connect_time_utc"])
        if sessions.empty:
            sessions = pd.DataFrame({
                "source_dataset": pd.Series(dtype=str),
                "entity_id": pd.Series(dtype=str),
                "connect_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            })
        return hourly, sessions

    def feature_rows(self):
        """One feature row per entity, for the most recent hour in the buffer."""
        hourly, sessions = self._frames()
        if hourly.empty:
            return None
        feats = P.build_feature_table(hourly, sessions)
        feats = feats.sort_values(["entity_id", "hour"])
        rows = feats.groupby("entity_id", as_index=False).tail(1).reset_index(drop=True)
        return self._restore_expanding(rows)

    def _restore_expanding(self, rows):
        """Overwrite expanding_mean with the true running value.

        pipeline computes it over whatever history it is handed; the ring buffer
        holds only KEEP_HOURS, so the value it derives is wrong. The statistic is
        the mean of every kWh observation strictly BEFORE the current hour, which
        is exactly what self.cum tracks.
        """
        if "expanding_mean" not in rows.columns:
            return rows
        vals = []
        for _, r in rows.iterrows():
            total, n = self.cum[r["entity_id"]]
            prior_n = n - 1                       # exclude the current hour
            prior_sum = total - float(r["kwh_allocated"])
            vals.append(prior_sum / prior_n if prior_n >= 6 else np.nan)
        rows = rows.copy()
        rows["expanding_mean"] = vals
        return rows

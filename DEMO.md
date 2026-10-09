# Demo guide — IRVE Load Simulator

## Run it

```bash
python3 -m uvicorn sim.server:app --port 8008
```

Open **http://localhost:8008**.

The server uses the saved Hurdle-XGBoost artifacts and six-agent MAPPO
checkpoint; it does not train either model at startup. For offline Paris
synthetic mode, choose **SYNTHETIC** in the UI.

**Start it ~10 minutes before you present.** The rolling metrics need 72
simulated hours to fill their window, and at 8s per simulated hour that is about
10 minutes of wall clock. A cold start shows "accumulating hours..." and a nearly
empty chart, which is a weak opening.

Check it is healthy without leaving your terminal:

```bash
curl -s localhost:8008/health
```

`"status":"running"` and a rising `sim_hours` means you are good. `"routed":true`
from `/api/meta` confirms street routing is active.

To stop: `Ctrl-C`, or `kill $(lsof -ti:8008)`.

---

## What is on screen

| Element | Meaning |
|---|---|
| Blue-ringed dot | The selected dataset's aggregate historical demand profile. |
| Small grey dots | The 6 stations selected from that location's OSM catalog. |
| Orange dots | Vehicles **driving to charge**. |
| Green dots | Vehicles **plugged in**. |
| Blue dots | Vehicles **finished, driving away**. |
| Faint orange lines | Active vehicle routes, following real streets. |
| Site colour | Live absolute error — green → amber → red. |
| Site halo size | Actual load that hour. |

Clock runs at **1 simulated hour per 8 seconds**, so a full day passes in about
3 minutes.

---

## Run the demo in this order

### 1. Open with the honesty, not the visuals (20 seconds)

Point at the two badges in the header before anything else.

> "The mode selector distinguishes real OpenStreetMap geography from the
> synthetic Paris fallback. The leakage-safe forecast badge means the model
> does not use sessions that start in the target hour. Instead it uses
> historical session arrivals from 24 hours earlier, plus historical demand
> lags and rolling statistics. So the inputs are available before the target
> hour begins."

Leading with this is the strongest move available. It means nobody in the room
gets to "catch" you with it later, and everything after it is trusted.

### 2. The map — what is actually being simulated (45 seconds)

> "The map follows the selected dataset: Paris, Boulder, Palo Alto, Dundee, or
> Perth. These are real OSM station locations; the six displayed stations are
> also the six MAPPO agents. Cars drive in, plug in, charge, and drive away."

Pan and zoom. It is a real map, and moving it makes that obvious. **Zoom in to
street level on a moving vehicle** - this is the detail worth showing. The cars
follow the actual road network, not straight lines between points:

> "Paris uses the cached OpenStreetMap road graph when available. Other
> locations use the simulator's approximate vehicle paths."

Then the key point about granularity:

> "The model forecast is at the dataset's aggregate profile level. We distribute
> it across the six OSM stations by capacity for control, but the historical
> corpus does not identify the actual demand of each physical charger."

### 3. The race against persistence — this is the actual argument (60 seconds)

Point at the two big numbers.

> "The headline isn't the model's error. It's the model's error **against
> persistence** — just predicting that this hour equals last hour."

Then explain why, because this is the most technically interesting thing you have:

> "Session energy is allocated proportionally across the hours a session spans.
> So every *interior* hour of a session gets exactly the same kWh. Aggregated
> across a site, this hour equals last hour **exactly** whenever no session
> starts or ends. With 4–8 hour connections, most consecutive hour-pairs are
> mechanical plateaus. Persistence is therefore a brutal baseline, and the
> model's real job is predicting the **transitions** — the starts and the stops."

The displayed comparison is a simulation diagnostic only; it is not a new
offline evaluation or an untouched-test result.

### 4. The chart — where it works and where it fails (45 seconds)

Point at the predicted-vs-actual chart.

> "Grey area is simulated actual load, blue is the model, orange is persistence.
> These live curves help inspect the simulator, not measure held-out research
> performance."

Then find a bad hour in the per-site table and say so out loud:

> "This is simulated/profile-resampled demand, not a held-out test set or
> station telemetry. The live error is a simulator diagnostic, not a research
> evaluation."

**Showing a failure deliberately is more convincing than hiding it.** It proves
you understand the model rather than just having plotted it.

### 5. Close on what makes it real (20 seconds)

> "Real station geography and historical demand are different data sources.
> The map shows OSM charge points, while the demand profile comes from the
> selected dataset's historical sessions. It is not observed station-specific
> demand."

---

## Questions you will get

**"Is this actually your model or a mock-up?"**
The existing saved XGBoost hurdle artifacts are served through
`sim.model_service.ModelService`; the simulation does not retrain them. You can
show the artifact metadata:
```bash
python3 -c "from sim.model_service import ModelService; m=ModelService(); print(len(m.columns),'features |',m.meta['data_source'])"
```

**"How do you know the live features match training?"**
```bash
python3 test_parity.py
```
Asserts the live ring buffer reproduces the offline feature builder to 1e-9. It earns its keep:
it caught `expanding_mean` drifting, because that feature expands over all
history and the live ring buffer only holds 336 hours. It's now carried as
explicit state.

**"Why only 6 stations?"**
The saved MAPPO checkpoint has six per-agent actors and a centralized critic
dimensioned for six agents. Reusing it without retraining requires six selected
stations. This is a controller/checkpoint limit, not a claim that the model
has station-specific historical demand.

**"Why is WAPE sometimes over 100%?"**
Overnight, actual load approaches zero and WAPE divides by it. Watch MAE and
RMSE at night, WAPE during the day. Say this before someone points at it.

**"Can you make it run faster / slower?"**
`SIM_HOUR_WALL_SECONDS` in `sim/engine.py`. Lower is faster.

**"Are the cars actually driving on roads?"**
Yes. The Paris road network is fetched once from Overpass, contracted to a
junction graph, and cached in `artifacts/roads.json`. Routing runs locally with
networkx at about 10ms per route. Rebuild with `python3 -m sim.roads`.

**"Is this SUMO?"**
No - it is a purpose-built agent simulation. SUMO models car-following physics at
the second level; this model predicts hourly energy per site, so sub-hour vehicle
dynamics land in the same bucket and would not move the numbers. What the model
needs is when cars arrive and how much they draw, which the agent layer produces
directly. SUMO would earn its place for battery-physics or congestion-feedback
claims.

**"Is this an untouched test evaluation?"**
No. The UI is a live simulation using historical dataset-level arrival/session
profiles and the saved model. Its rolling errors are not a new untouched-test
evaluation and must not be reported as research results.

---

## Do not claim

- That live simulator errors are research results; the values come from a
  simulated rollout, not an untouched test-set evaluation.
- That the model uses a leakage-safe historical arrival feature. It no longer uses
  `sessions_started(t)`; by default it uses `sessions_started_lag24h` to predict
  the target hour. The current-hour session count is excluded from model inputs.
- That each mapped OSM station has its own observed historical demand. The
  aggregate forecast and historical session profile are assigned across the
  six controlled stations; no physical station-demand correspondence is
  present in the datasets.
- That the model beats persistence **on the real data**. That comparison has not
  been run — the simulation computes persistence live, but there is no offline
  baseline in the notebooks yet.

## If something breaks

| Symptom | Cause |
|---|---|
| "accumulating hours…" | Normal for the first ~6 simulated hours. Wait. |
| Blank map, rail works | IGN tiles blocked — check network. The sim itself is fine. |
| `FileNotFoundError: artifacts` | Run `python3 train_for_sim.py` first. |
| Port in use | `kill $(lsof -ti:8008)` |
| Cars move in straight lines | Road cache missing — run `python3 -m sim.roads` |
| Page frozen, no errors | Check `curl -s localhost:8008/health` for `status` and `error` |
| Empty map, everything 0.0 | You started at 03:00 sim time. Wait for morning, or restart. |
| Layout stacked / cramped | Browser window under 900px wide. Widen it. |

# Station-level energy-management environment

This numerical MARL environment gives one agent control of each configured
station. It does not depend on the map. MAPPO training is a separate module and
does not change the physical environment or forecasting pipeline.

## Units and timestep

The environment timestep is **one hour**. Every energy quantity is consistently
expressed as **kWh per timestep**:

- Demand: kWh/timestep
- Renewable availability: kWh/timestep
- Total grid capacity and current grid load: kWh/timestep
- Remaining grid capacity: kWh/timestep
- Station capacity: kWh/timestep (an interval energy limit, not instantaneous
  kW charging power)
- Allocated, served, unused, and unserved energy: kWh/timestep

`current_grid_load` is energy already consumed within the current one-hour grid
budget. The allocation budget is
`max(total_grid_capacity - current_grid_load, 0)`.

## Basic use

```python
from marl import EnergyManagementEnv, MARLConfig
from marl.baseline import proportional_allocation

config = MARLConfig(
    station_ids=("A", "B"),
    station_capacities={"A": 100.0, "B": 200.0},
    renewable_availability=50.0,
    grid_capacity=200.0,
)
env = EnergyManagementEnv(config)
forecast = {"A": 80.0, "B": 150.0}
observations, reset_info = env.reset(
    current_demand={"A": 75.0, "B": 140.0},
    forecast=forecast,
    current_grid_load=20.0,
    forecast_source="model",
)
action = proportional_allocation(
    forecast,
    station_capacities=config.station_capacities,
    renewable_available=config.renewable_availability,
    grid_capacity=config.grid_capacity,
    current_grid_load=20.0,
)
rewards, terminated, info = env.step(
    action,
    forecast=forecast,
    actual_demand_next=lambda: {"A": 90.0, "B": 135.0},
    forecast_source="model",
)
```

Agents observe forecast demand at `t+1`, current demand at `t`, station
capacity, renewable availability, remaining grid capacity, and current grid
load. `normalize_observations=True` scales resources using the values active in
the current reset, including per-reset overrides.

Actions request renewable and grid allocations per station. The environment
clips each combined station allocation to its forecast demand and capacity,
then proportionally projects renewable and grid requests onto their respective
available budgets. Attempted excess requests are reported separately from
feasible allocations.

The actual next-demand provider must be a callable. It is invoked only after
the action has been projected. Reward penalizes **realized** unserved demand.
Forecast-unserved demand is reported separately. Actual energy served is capped
at actual demand; excess allocation is reported as unused/curtailed. Grid cost
uses allocated grid energy; the renewable reward uses renewable energy
actually used to serve demand.

## Forecast provenance

`PersistenceForecast` and `ModelForecast` are normal forecast sources.
`OracleForecast` is an evaluation upper-bound source only. Set
`allow_oracle_forecast=True` in `MARLConfig` only for that explicit experiment;
the environment otherwise rejects `forecast_source="oracle"`. The caller must
provide the matching source name to `reset()` and `step()`.

## Baseline

`proportional_allocation()` takes the same forecast, station capacities,
renewable availability, total grid capacity, and current grid load as the
environment. It allocates renewables first and then grid energy, with
demand-proportional redistribution around station capacity limits.

## Demo and tests

After generating compatible simulator artifacts:

```powershell
python -m marl.demo_environment
python -m pytest marl/tests -q
```

The demo is a **Forecast-driven MARL environment demonstration**, not a trained
MARL policy. It obtains model forecasts, selects a proportional action, and
retrieves synthetic next-hour actual demand lazily after action projection.

## MAPPO training

Install project dependencies, then run the persistence-forecast training
example:

```powershell
python -m marl.train_mappo --episodes 200 --steps-per-episode 128 --seed 42
```

The default run uses the six demo stations, independent local actors, and a
centralized critic. Each actor is queried with only its own local observation;
the global concatenated observation is supplied only to the training critic.
`ForecastSource` adapters are supplied to the trainer; the
training API accepts either persistence or a caller-provided model forecast
adapter without importing forecasting internals. The CLI intentionally uses
persistence by default. Oracle forecasts are rejected for training.

The evaluator can run the trained policy deterministically and compare it with
the proportional baseline over the same scenario objects and forecast source.
This is an evaluation interface, not evidence that MAPPO outperforms the
baseline. Evaluation reports `peak_realized_grid_usage` from realized grid
energy used to serve demand, rather than grid energy merely allocated.

Checkpoints and run metadata are written beneath `artifacts/marl/`:
`mappo_final.pt`, periodic `mappo_episode_*.pt`, actor and critic state files,
and `training_config.json`. For a quick interaction/gradient/checkpoint smoke
test, run:

```powershell
python -m pytest marl/tests/test_mappo.py -q
```

## Existing hurdle-model forecast → MAPPO experiment

The separate `marl.hurdle_experiment` runner loads the existing synthetic-trained
hurdle-model artifacts and warms its causal feature buffer from the saved
hourly/session history. It generates future demand from the same session
arrival and charging-energy process used for the model's synthetic training
data. Future actual demand remains hidden from the actor until after its action;
evaluation replays each MAPPO forecast path for the proportional baseline.
This experiment uses the six Paris demo entities supported by the saved model
and writes to a separate `artifacts/marl/hurdle_mappo/` directory.

The planned training command is:

```powershell
python -m marl.hurdle_experiment --seed 42 --episodes 200 --steps-per-episode 128 --evaluation-interval 10 --evaluation-episodes 20 --output-directory artifacts/marl/hurdle_mappo
```

This verifies experiment plumbing only; completing a run does not establish
that MAPPO outperforms proportional allocation.

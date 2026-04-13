# Start-State Audit — env_11 Initial Thermal Conditions
**Date:** 2026-04-12  
**File audited:** `src/environment/aluminum_melting_env_11.py`

---

## State Vector Layout (8-element, returned by reset() and step())

| Index | Variable | Unit | Cold-start | Hot-start (idle=0) | Hot-start (idle=30) |
|-------|----------|------|-----------|-------------------|---------------------|
| [0] | `temperature` (metal) | °C | **25.0** | **165.0** | 109.7 |
| [1] | `weight` | kg | 350.0 | 350.0 | 350.0 |
| [2] | `time` | sec | 0.0 | 0.0 | 0.0 |
| [3] | `power` | kW | 0.0 | 0.0 | 0.0 |
| [4] | `status` | int | 1 (ON) | 1 (ON) | 1 (ON) |
| [5] | `energy_consumption` | kWh | 0.0 | 0.0 | 0.0 |
| [6] | `scrap_added` | kg | 0.0 | 0.0 | 0.0 |
| [7] | `furnace_wall_temp` | °C | **25.0** | **225.0** | 146.2 |

---

## Constructor Parameters for Start-State Control

| Parameter | Default | Effect |
|-----------|---------|--------|
| `start_mode` | `"hot"` | `"cold"` → all temps = 25°C ambient; `"hot"` → exponential decay |
| `idle_time_min` | `0` | Minutes since last batch; controls residual heat in hot mode |
| `wall_temp_c` | `None` | Direct override for wall temperature (bypasses formula) |
| `initial_metal_temp_c` | `None` | Direct override for metal temperature |

---

## Residual Heat Formula (hot-start mode)

```
hot_wall_delta_c   = 200.0  (max residual heat at idle=0)
hot_wall_tau_min   = 60.0   (decay time constant, minutes)
hot_metal_fraction = 0.70   (metal temp as fraction of wall residual)

wall_temp  = 25 + 200 × exp(−idle_time_min / 60)
metal_temp = 25 + 0.70 × (wall_temp − 25)
```

| idle_time_min | wall_temp (°C) | metal_temp (°C) |
|---------------|----------------|-----------------|
| 0 | 225.0 | 165.0 |
| 10 | 181.3 | 136.9 |
| 20 | 154.5 | 117.2 |
| 30 | 146.2 | 109.7 |
| 60 | 98.6 | 76.1 |
| 120 | 52.1 | 39.5 |
| ∞ (cold) | 25.0 | 25.0 |

---

## Current Evaluation Protocol (Phase 1)

| Script | start_mode | idle_time_min | Scenarios covered |
|--------|-----------|---------------|-------------------|
| `eval_rl_extended.py` | `"hot"` | Uniform(0, 30) | Hot-start only |
| `eval_rl_safety.py` | `"hot"` | Uniform(0, 30) | Hot-start only |
| `ablation_batch_level.py` | `"hot"` | Uniform(0, 30) | Hot-start only |

**All Phase 1 evaluations assumed hot-start.** Wall temp range covered: 146–225°C.

---

## Gap vs. Plant Baseline (N=103)

The 103-batch plant dataset (energy 496–735 kWh, duration 87–129 min) has substantially
wider spread than simulated hot-start episodes (566–631 kWh, 87–97 min). The wide range
and strong duration–energy correlation (r=0.745) indicate real plant batches include
varying start-state conditions (cold-start for first batch of shift, hot-start intra-shift).

Cold-start events in real plant are expected for:
- First batch after overnight idle
- First batch after maintenance shutdown
- Batches following extended breaks (>2 hours)

Hot/cold labels are NOT available in plant records. Phase 2 uses scenario-based evaluation
with documented assumptions.

---

## Safety Override (line 440–443)

```python
is_alloying_time = 30.0 <= current_minute < 31.0
if is_alloying_time:
    self.state["power"] = 0.0  # Hard override, always active
```

This override is independent of start-state — applies identically in cold and hot episodes.

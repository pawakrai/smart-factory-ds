if __package__:
    from . import app_v9 as sim
else:
    import app_v9 as sim


def _choose_idle_furnace(state):
    idle = list(state.get("idle_furnaces", []))
    if not idle:
        return None
    last = state.get("last_started_furnace")
    if last in idle and len(idle) > 1:
        for f in idle:
            if f != last:
                return f
    return idle[0]


def continuous_melting_controller(state):
    """Baseline: keep melting whenever possible with minimal gating."""
    if state.get("remaining_batches", 0) <= 0:
        return {"start_allowed": False, "delay_reason": "no_batches_left"}
    chosen_if = _choose_idle_furnace(state)
    if chosen_if is None:
        return {"start_allowed": False, "delay_reason": "no_idle_furnace"}
    if state.get("any_holding_active", False) and (not sim.ALLOW_PARALLEL_IF):
        return {"start_allowed": False, "delay_reason": "holding_guard"}
    return {
        "start_allowed": True,
        "chosen_if": int(chosen_if),
        "selected_power": 500.0,
        "force_mode_override": False,
        "bypass_jit_gate": True,
        "bypass_demand_guard": True,
        "delay_reason": None,
    }


def make_rule_based_controller(
    price_threshold=3.2,
    peak_headroom_kw=120.0,
    fixed_power=475.0,
    urgency_start_threshold=0.60,
):
    """Rule-based controller with JIT, peak headroom and TOU awareness."""

    def _controller(state):
        if state.get("remaining_batches", 0) <= 0:
            return {"start_allowed": False, "delay_reason": "no_batches_left"}

        chosen_if = _choose_idle_furnace(state)
        if chosen_if is None:
            return {"start_allowed": False, "delay_reason": "no_idle_furnace"}

        if state.get("any_holding_active", False) and (not sim.ALLOW_PARALLEL_IF):
            return {"start_allowed": False, "delay_reason": "holding_guard"}

        force_mode = bool(state.get("force_mode", False))
        policy_state = state.get("policy_state", {})
        depletion_urgency = float(policy_state.get("depletion_urgency", 0.0))
        tou_price = float(state.get("tou_effective_price", 0.0))

        projected_kw = (
            float(state.get("baseline_kw", 0.0))
            + float(state.get("if_kw_now", 0.0))
            + float(fixed_power)
        )
        max_allow = float(sim.CONTRACT_DEMAND_KW - peak_headroom_kw)
        if (not force_mode) and projected_kw > max_allow:
            return {"start_allowed": False, "delay_reason": "peak_headroom_guard"}

        if (
            (not force_mode)
            and tou_price > float(price_threshold)
            and depletion_urgency < float(urgency_start_threshold)
        ):
            return {"start_allowed": False, "delay_reason": "tou_avoidance"}

        return {
            "start_allowed": True,
            "chosen_if": int(chosen_if),
            "selected_power": float(fixed_power),
            "force_mode_override": force_mode,
            "delay_reason": None,
        }

    _controller.__name__ = "rule_based_controller"
    return _controller


def rule_based_controller(state):
    """Default rule-based controller entrypoint (with default thresholds)."""
    return make_rule_based_controller()(state)

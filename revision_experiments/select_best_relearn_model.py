"""
select_best_relearn_model.py — Predefined Model Selection & Final Reports
===========================================================================
Applies the predefined selection rule to evaluation results from the
reward-retuning study and produces:
  - best_model_selection.md
  - relearn_results_summary.md

Selection rule (predefined, no post-hoc changes):
  1. Discard success_rate < 100% under mixed_80_20
  2. Discard executed_violations > 0 under any scenario
  3. Discard duration > original DQN mixed_80_20 + 2.0 min
  4. Rank remaining by lowest energy_mean under mixed_80_20
  5. Tiebreak: lower overshoot → shorter duration

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/select_best_relearn_model.py
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from revision_experiments.relearn_config import (
    OUTPUT_ROOT, DURATION_GUARD_MINUTES, git_commit_short,
)

EVAL_DIR = OUTPUT_ROOT / "eval"
BASELINES = ["dqn_final", "expert_profile", "always_max_450kw"]


def main():
    print("=" * 70)
    print("select_best_relearn_model.py — Model Selection")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    sum_path = EVAL_DIR / "all_candidates_summary.csv"
    ep_path = EVAL_DIR / "all_candidates_per_episode.csv"

    if not sum_path.exists() or not ep_path.exists():
        print(f"[ERROR] Evaluation CSVs not found in {EVAL_DIR}")
        print("        Run eval_rl_relearn_grid.py first.")
        return

    df_sum = pd.read_csv(sum_path)
    df_ep = pd.read_csv(ep_path)

    # ── Reference: original DQN final under mixed_80_20 ─────────────────────
    dqn_mixed = df_sum[(df_sum["policy"] == "dqn_final") & (df_sum["scenario"] == "mixed_80_20")]
    if dqn_mixed.empty:
        print("[ERROR] dqn_final mixed_80_20 results not found.")
        return
    dqn_ref = dqn_mixed.iloc[0]
    dqn_energy = dqn_ref["energy_mean"]
    dqn_duration = dqn_ref["duration_mean"]
    dqn_overshoot = dqn_ref["overshoot_mean"]
    max_duration = dqn_duration + DURATION_GUARD_MINUTES

    print(f"\nReference (dqn_final mixed_80_20):")
    print(f"  Energy   : {dqn_energy:.2f} kWh")
    print(f"  Duration : {dqn_duration:.1f} min")
    print(f"  Overshoot: {dqn_overshoot:.2f} °C")
    print(f"  Max allowed duration: {max_duration:.1f} min (+{DURATION_GUARD_MINUTES} guard)")

    # ── Identify candidates (exclude baselines) ─────────────────────────────
    all_policies = df_sum["policy"].unique()
    candidate_policies = [p for p in all_policies if p not in BASELINES]
    print(f"\nCandidates: {len(candidate_policies)}")

    # ── Apply selection gates ────────────────────────────────────────────────
    gate_results = []

    for pol in sorted(candidate_policies):
        mixed = df_sum[(df_sum["policy"] == pol) & (df_sum["scenario"] == "mixed_80_20")]
        if mixed.empty:
            continue
        row = mixed.iloc[0]

        # Gate 1: success_rate == 100%
        gate1_pass = row["success_rate"] >= 1.0

        # Gate 2: executed_violations == 0 under ALL scenarios
        all_scenarios = df_sum[df_sum["policy"] == pol]
        gate2_pass = all_scenarios["executed_violations_mean"].max() == 0.0

        # Gate 3: duration guard
        gate3_pass = row["duration_mean"] <= max_duration

        passed = gate1_pass and gate2_pass and gate3_pass

        gate_results.append({
            "policy": pol,
            "energy_mean": row["energy_mean"],
            "energy_sd": row["energy_sd"],
            "duration_mean": row["duration_mean"],
            "duration_sd": row["duration_sd"],
            "success_rate": row["success_rate"],
            "overshoot_mean": row["overshoot_mean"],
            "executed_violations_max": all_scenarios["executed_violations_mean"].max(),
            "gate1_success": gate1_pass,
            "gate2_violations": gate2_pass,
            "gate3_duration": gate3_pass,
            "passed_all": passed,
        })

    df_gates = pd.DataFrame(gate_results)

    # ── Rank passing candidates ──────────────────────────────────────────────
    passing = df_gates[df_gates["passed_all"]].copy()

    if passing.empty:
        print("\n[RESULT] No candidates passed all gates.")
        print("         The original DQN model should be retained.")
        winner = None
        winner_row = None
    else:
        passing = passing.sort_values(
            by=["energy_mean", "overshoot_mean", "duration_mean"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        passing["rank"] = range(1, len(passing) + 1)

        winner = passing.iloc[0]["policy"]
        winner_row = passing.iloc[0]
        print(f"\n[RESULT] Best candidate: {winner}")
        print(f"  Energy   : {winner_row['energy_mean']:.2f} ± {winner_row['energy_sd']:.2f} kWh")
        print(f"  Duration : {winner_row['duration_mean']:.1f} ± {winner_row['duration_sd']:.1f} min")
        print(f"  Overshoot: {winner_row['overshoot_mean']:.2f} °C")
        energy_diff = winner_row["energy_mean"] - dqn_energy
        print(f"  Δ energy vs original: {energy_diff:+.2f} kWh ({energy_diff/dqn_energy*100:+.2f}%)")

    # ── Write best_model_selection.md ────────────────────────────────────────
    _write_selection_report(df_gates, passing, winner, dqn_ref, max_duration)

    # ── Write relearn_results_summary.md ─────────────────────────────────────
    _write_results_summary(df_sum, df_ep, df_gates, passing, winner, dqn_ref)

    print(f"\n{'='*70}")
    print("Selection complete.")
    print(f"  {OUTPUT_ROOT / 'best_model_selection.md'}")
    print(f"  {OUTPUT_ROOT / 'relearn_results_summary.md'}")
    print(f"{'='*70}")


def _write_selection_report(df_gates, passing, winner, dqn_ref, max_duration):
    lines = [
        "# Best Model Selection Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Git commit**: {git_commit_short()}",
        "",
        "## Selection Rule (Predefined)",
        "",
        "1. Discard `success_rate < 100%` under mixed_80_20",
        "2. Discard `executed_violations > 0` under any scenario",
        f"3. Discard `duration_mean > {max_duration:.1f} min` (original DQN + {DURATION_GUARD_MINUTES} min guard)",
        "4. Rank remaining by lowest `energy_mean` under mixed_80_20",
        "5. Tiebreak: lower overshoot_mean → shorter duration_mean",
        "",
        f"## Reference: Original DQN Final (mixed_80_20)",
        "",
        f"- Energy: {dqn_ref['energy_mean']:.2f} ± {dqn_ref['energy_sd']:.2f} kWh",
        f"- Duration: {dqn_ref['duration_mean']:.1f} ± {dqn_ref['duration_sd']:.1f} min",
        f"- Overshoot: {dqn_ref['overshoot_mean']:.2f} °C",
        "",
        "## Gate Results",
        "",
        "| Candidate | Energy (kWh) | Duration (min) | Success | Violations | Gate1 | Gate2 | Gate3 | Pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for _, r in df_gates.iterrows():
        lines.append(
            f"| {r['policy']} | {r['energy_mean']:.2f} ± {r['energy_sd']:.2f} | "
            f"{r['duration_mean']:.1f} ± {r['duration_sd']:.1f} | "
            f"{r['success_rate']:.0%} | {r['executed_violations_max']:.0f} | "
            f"{'PASS' if r['gate1_success'] else 'FAIL'} | "
            f"{'PASS' if r['gate2_violations'] else 'FAIL'} | "
            f"{'PASS' if r['gate3_duration'] else 'FAIL'} | "
            f"{'**PASS**' if r['passed_all'] else 'FAIL'} |"
        )

    if not passing.empty:
        lines.extend([
            "",
            "## Ranked Candidates (Passing All Gates)",
            "",
            "| Rank | Candidate | Energy (kWh) | Overshoot (°C) | Duration (min) | Δ Energy |",
            "|---|---|---|---|---|---|",
        ])
        for _, r in passing.iterrows():
            delta = r["energy_mean"] - dqn_ref["energy_mean"]
            lines.append(
                f"| {r['rank']} | {r['policy']} | {r['energy_mean']:.2f} ± {r['energy_sd']:.2f} | "
                f"{r['overshoot_mean']:.2f} | {r['duration_mean']:.1f} | {delta:+.2f} kWh |"
            )
        lines.extend([
            "",
            f"## Winner: `{winner}`",
            "",
            f"Selected per predefined rule. No manual override applied.",
        ])
    else:
        lines.extend([
            "",
            "## Result: No Improvement",
            "",
            "No candidates passed all selection gates.",
            "The original DQN final model should be retained.",
        ])

    (OUTPUT_ROOT / "best_model_selection.md").write_text("\n".join(lines), encoding="utf-8")


def _write_results_summary(df_sum, df_ep, df_gates, passing, winner, dqn_ref):
    lines = [
        "# RL Reward-Retuning Results Summary",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Git commit**: {git_commit_short()}",
        "",
        "## Study Design",
        "",
        "- **Objective**: Test whether increasing the energy penalty coefficient and/or adding",
        "  an overshoot penalty can reduce mean batch energy under the fixed evaluation protocol.",
        "- **Training**: start_mode='hot' only (preserving original training distribution).",
        "  Generalization assessed via cold-start and mixed-start evaluation.",
        "- **Evaluation**: 100 episodes × 3 scenarios (cold_start, hot_start, mixed_80_20),",
        "  seed=2024, identical protocol to eval_rl_cold_hot_mixed.py.",
        "- **Selection**: Predefined rule — 100% success, 0 executed violations,",
        f"  duration guard ({DURATION_GUARD_MINUTES} min), rank by lowest mixed_80_20 energy.",
        "",
        "## Comparison Table (mixed_80_20)",
        "",
        "| Policy | Energy (kWh) | SD | Duration (min) | SD | Success | Overshoot (°C) |",
        "|---|---|---|---|---|---|---|",
    ]

    # Show baselines + all candidates under mixed_80_20
    mixed = df_sum[df_sum["scenario"] == "mixed_80_20"].sort_values("energy_mean")
    for _, r in mixed.iterrows():
        tag = ""
        if r["policy"] == winner:
            tag = " **← BEST**"
        elif r["policy"] == "dqn_final":
            tag = " (original)"
        lines.append(
            f"| {r['policy']}{tag} | {r['energy_mean']:.2f} | {r['energy_sd']:.2f} | "
            f"{r['duration_mean']:.1f} | {r['duration_sd']:.1f} | "
            f"{r['success_rate']:.0%} | {r['overshoot_mean']:.2f} |"
        )

    # Expert and always-max references
    expert_mixed = df_sum[(df_sum["policy"] == "expert_profile") & (df_sum["scenario"] == "mixed_80_20")]
    amax_mixed = df_sum[(df_sum["policy"] == "always_max_450kw") & (df_sum["scenario"] == "mixed_80_20")]

    lines.extend(["", "## Key Findings", ""])

    if winner is not None:
        w_row = passing.iloc[0]
        delta_e = w_row["energy_mean"] - dqn_ref["energy_mean"]
        delta_d = w_row["duration_mean"] - dqn_ref["duration_mean"]
        pct_e = delta_e / dqn_ref["energy_mean"] * 100

        # Compare to expert
        expert_e = expert_mixed.iloc[0]["energy_mean"] if not expert_mixed.empty else None
        delta_vs_expert = w_row["energy_mean"] - expert_e if expert_e else None

        lines.append(f"### 1. Energy Change")
        lines.append(f"- Revised RL (`{winner}`): {w_row['energy_mean']:.2f} ± {w_row['energy_sd']:.2f} kWh")
        lines.append(f"- Original DQN: {dqn_ref['energy_mean']:.2f} ± {dqn_ref['energy_sd']:.2f} kWh")
        lines.append(f"- **Δ energy: {delta_e:+.2f} kWh ({pct_e:+.2f}%)**")

        if abs(delta_e) <= 1.0:
            lines.append(f"- Interpretation: No meaningful improvement (Δ ≤ 1 kWh).")
        elif delta_e < -1.0:
            if abs(delta_e) < 5.0:
                lines.append(f"- Interpretation: Slight improvement.")
            else:
                lines.append(f"- Interpretation: Notable improvement.")
        else:
            lines.append(f"- Interpretation: Energy increased — no improvement.")

        lines.append("")
        lines.append("### 2. Duration Change")
        lines.append(f"- Revised RL: {w_row['duration_mean']:.1f} ± {w_row['duration_sd']:.1f} min")
        lines.append(f"- Original DQN: {dqn_ref['duration_mean']:.1f} ± {dqn_ref['duration_sd']:.1f} min")
        lines.append(f"- Δ duration: {delta_d:+.1f} min")
        if abs(delta_d) <= 1.0:
            lines.append(f"- Duration is materially unchanged.")
        elif delta_d > 1.0:
            lines.append(f"- Duration increased by {delta_d:.1f} min (within {DURATION_GUARD_MINUTES}-min guard).")

        lines.append("")
        lines.append("### 3. Safety")
        lines.append(f"- Executed violations: 0 across all scenarios (env safety override confirmed)")
        lines.append(f"- Success rate: 100% (all batches reached target temperature)")

        if delta_vs_expert is not None:
            lines.append("")
            lines.append("### 4. Comparison vs Expert Profile")
            lines.append(f"- Expert energy (mixed_80_20): {expert_e:.2f} kWh")
            lines.append(f"- Revised RL vs expert: {delta_vs_expert:+.2f} kWh")
            if delta_vs_expert < -1.0:
                lines.append(f"- **Revised RL beats expert profile.**")
            elif abs(delta_vs_expert) <= 1.0:
                lines.append(f"- Revised RL matches expert profile (within 1 kWh).")
            else:
                lines.append(f"- Revised RL remains slightly above expert profile.")

        lines.extend([
            "",
            "### 5. Manuscript Recommendation",
            "",
        ])
        if delta_e < -1.0:
            lines.append(
                f"The revised RL controller (`{winner}`) achieved a {abs(delta_e):.2f} kWh "
                f"({abs(pct_e):.2f}%) energy reduction under the same evaluation protocol. "
                f"This is a controlled, reproducible improvement and is suitable for reporting "
                f"in the revised manuscript."
            )
        elif abs(delta_e) <= 1.0:
            lines.append(
                f"The reward-retuning study did not yield a meaningful energy improvement "
                f"(Δ = {delta_e:+.2f} kWh). The original DQN result should be retained in "
                f"the manuscript. The study demonstrates that the RL controller's performance "
                f"is robust to moderate reward perturbations."
            )
        else:
            lines.append(
                f"The best retuned candidate had higher energy than the original DQN. "
                f"The original result should be retained."
            )
    else:
        lines.extend([
            "### No candidates passed all selection gates.",
            "",
            "The original DQN final model should be retained in the manuscript.",
            "The retuning study demonstrates that the current reward design is stable —",
            "moderate reward perturbations do not yield meaningful improvements, and",
            "aggressive perturbations may violate success or duration constraints.",
        ])

    lines.extend([
        "",
        "## Reproducibility",
        "",
        "- Training seeds: [42, 2024, 7]",
        "- Evaluation seed: 2024",
        "- Training episodes: 1500 per config per seed",
        "- Evaluation episodes: 100 per scenario per policy",
        "- All models saved to: outputs/revision_relearn/models/",
        "- Per-episode data: outputs/revision_relearn/eval/all_candidates_per_episode.csv",
        "- Manifests: outputs/revision_relearn/relearn_manifest.json, eval/eval_manifest.json",
    ])

    (OUTPUT_ROOT / "relearn_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

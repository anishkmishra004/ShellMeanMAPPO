"""Stateful simulation service wrapping MultiShellLEOEnv for Streamlit session management."""
from __future__ import annotations
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from environment.leo_env import MultiShellLEOEnv
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.rule_based_policy import RuleBasedPolicy
from evaluation.evaluate import run_episode, evaluate_policy

POLICY_NAMES = ["Rule-Based Conflict-Aware", "Greedy", "Random", "ShellMean-MAPPO"]


def _ecef_to_latlon(pos_km: np.ndarray) -> Tuple[float, float]:
    """
    Convert ECEF position (km) to geographic (lat_deg, lon_deg).

    Args:
        pos_km: [x, y, z] in km.

    Returns:
        Tuple of (latitude_deg, longitude_deg).
    """
    x, y, z = pos_km
    r = float(np.linalg.norm(pos_km))
    if r < 1.0:
        return 0.0, 0.0
    lat = float(np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    lon = float(np.degrees(np.arctan2(y, x)))
    return lat, lon


def build_policy(policy_name: str, config: Dict[str, Any], seed: int) -> Any:
    """
    Instantiate a policy by name.

    Args:
        policy_name: One of POLICY_NAMES.
        config: Configuration dictionary.
        seed: Random seed.

    Returns:
        Policy instance with select_actions() or step_evaluation() method.
    """
    if policy_name == "Random":
        return RandomPolicy(config, seed=seed)
    elif policy_name == "Greedy":
        return GreedyPolicy(config, seed=seed)
    elif policy_name == "Rule-Based Conflict-Aware":
        return RuleBasedPolicy(config, seed=seed)
    elif policy_name == "ShellMean-MAPPO":
        from training.train import MAPPOTrainer
        from training.checkpoint import load_checkpoint
        trainer = MAPPOTrainer(config)
        chk = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
        chk = os.path.normpath(chk)
        if os.path.exists(chk):
            try:
                load_checkpoint(trainer.agent, chk)
            except Exception:
                # If network dimension changed from custom preset, proceed with initialized agent
                pass
        return trainer
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def call_policy(policy: Any, observations: Dict, masks: Dict, env: Optional[Any] = None) -> Dict:
    """Unified policy call regardless of interface type."""
    if hasattr(policy, "select_actions"):
        try:
            return policy.select_actions(observations, masks, env=env)
        except TypeError:
            return policy.select_actions(observations, masks)
    elif hasattr(policy, "step_evaluation"):
        return policy.step_evaluation(observations, masks)
    raise ValueError(f"Policy {type(policy)} has no select_actions or step_evaluation method.")


class SimulationSession:
    """
    Manages a single simulation session: environment, policy, and telemetry history.
    Designed for Streamlit session_state storage.
    """

    def __init__(self, config: Dict[str, Any], policy_name: str = "Rule-Based Conflict-Aware", seed: int = 42):
        self.config = config
        self.policy_name = policy_name
        self.seed = seed
        self.env = MultiShellLEOEnv(config, seed=seed)
        self.policy = build_policy(policy_name, config, seed)
        self.observations: Optional[Dict] = None
        self.current_step: int = 0
        self.is_initialized: bool = False
        self.is_done: bool = False
        self.telemetry: List[Dict[str, Any]] = []
        self.event_log: List[str] = []

    @property
    def current_slot(self) -> int:
        """Alias for current_step to support slot terminology."""
        return self.current_step

    @current_slot.setter
    def current_slot(self, value: int) -> None:
        self.current_step = value

    def set_policy(self, policy_name: str) -> None:
        """Change policy dynamically."""
        self.policy_name = policy_name
        self.policy = build_policy(policy_name, self.config, self.seed)

    def initialize(self) -> Dict[str, Any]:
        """
        Reset the environment and return initial state info.

        Returns:
            Dictionary of initial state summary.
        """
        self.observations, info = self.env.reset(seed=self.seed)
        if hasattr(self.policy, "reset"):
            self.policy.reset(seed=self.seed)
        self.current_step = 0
        self.is_initialized = True
        self.is_done = False
        self.telemetry.clear()
        self.event_log.clear()

        self.event_log.append(f"[Slot 0] Environment initialized. Seed={self.seed}.")
        queues = self.env.traffic_model.get_queues()
        self.event_log.append(
            f"[Slot 0] Total initial backlog: {float(np.sum(queues)):.1f} Mb across "
            f"{self.env.num_ground_cells} ground cells."
        )
        self.event_log.append(
            f"[Slot 0] Policy: {self.policy_name}. "
            f"Constellation: {self.env.num_satellites} satellites, {self.env.num_shells} shells."
        )
        return info

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Execute one simulation slot.

        Returns:
            Telemetry dictionary for this slot, or None if already done.
        """
        if not self.is_initialized or self.is_done:
            return None

        masks = self.env.get_action_masks()
        t0 = time.perf_counter()
        actions = call_policy(self.policy, self.observations, masks, env=self.env)
        policy_time_ms = (time.perf_counter() - t0) * 1000.0

        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.current_step += 1

        self._build_event_log(info, actions)
        tel = self._build_telemetry(info, policy_time_ms)
        self.telemetry.append(tel)

        self.observations = next_obs
        self.is_done = terminated or truncated
        return tel

    def step_n(self, n: int) -> List[Dict[str, Any]]:
        """Execute n simulation slots sequentially."""
        records = []
        for _ in range(n):
            if self.is_done:
                break
            rec = self.step()
            if rec is not None:
                records.append(rec)
        return records

    def _build_event_log(self, info: Dict, actions: Dict) -> None:
        """Generate event log entries from actual simulation data."""
        slot = self.current_step
        outcome = self.env.last_resolution_outcome

        if outcome:
            for req in outcome.retained_requests:
                if req.cell_id > 0:
                    score = outcome.resolution_scores.get((req.sat_id, req.beam_id), 0.0)
                    self.event_log.append(
                        f"[Slot {slot:03d}] Retained: Sat S{req.sat_id:02d} Beam B{req.beam_id} -> Cell C{req.cell_id:02d} (Score: {score:.3f})"
                    )

            for req in outcome.rejected_requests:
                if req.cell_id > 0:
                    reason = outcome.rejection_reasons.get((req.sat_id, req.beam_id), "contention limit exceeded")
                    self.event_log.append(
                        f"[Slot {slot:03d}] Rejected: Sat S{req.sat_id:02d} Beam B{req.beam_id} -> Cell C{req.cell_id:02d} ({reason})"
                    )

        self.event_log.append(
            f"[Slot {slot:03d}] Summary: Served {info.get('served_traffic_mb', 0):.2f} Mb | Backlog {info.get('remaining_backlog_mb', 0):.1f} Mb | SINR {info.get('mean_sinr_db', 0):.1f} dB"
        )

        if len(self.event_log) > 600:
            self.event_log = self.event_log[-400:]

    def _build_telemetry(self, info: Dict, policy_time_ms: float) -> Dict[str, Any]:
        """Build a structured telemetry record for this slot."""
        tel: Dict[str, Any] = {
            "slot": self.current_step,
            "team_reward": float(info.get("team_reward", 0.0)),
            "served_traffic_mb": float(info.get("served_traffic_mb", 0.0)),
            "arrivals_mb": float(info.get("arrivals_mb", 0.0)),
            "remaining_backlog_mb": float(info.get("remaining_backlog_mb", 0.0)),
            "num_retained": int(info.get("num_retained", 0)),
            "num_rejected": int(info.get("num_rejected", 0)),
            "interference_watts": float(info.get("interference_watts", 0.0)),
            "jains_fairness": float(info.get("jains_fairness", 1.0)),
            "switching_count": int(info.get("switching_count", 0)),
            "mean_sinr_db": float(info.get("mean_sinr_db", 0.0)),
            "policy_time_ms": float(policy_time_ms),
        }

        beam_data = []
        trans = self.env.last_transmission_result
        if trans:
            for (sat_id, beam_id), bm in trans.beam_metrics.items():
                beam_data.append({
                    "sat_id": sat_id,
                    "beam_id": beam_id,
                    "cell_id": bm.cell_id,
                    "is_retained": bm.is_retained,
                    "is_active": bm.is_active,
                    "tx_power_w": float(bm.tx_power_w),
                    "sinr_db": float(bm.sinr_db),
                    "rate_mbps": float(bm.rate_mbps),
                    "delivered_mb": float(bm.served_data_mb),
                    "interference_w": float(bm.interference_w),
                    "slant_range_km": float(self.env.channel_model.compute_slant_range(self.env.satellite_network.satellites[sat_id].position, self.env.traffic_model.cells[bm.target_cell_index].position)) if (bm.target_cell_index is not None and bm.target_cell_index < len(self.env.traffic_model.cells)) else float(getattr(bm, "slant_range_km", 0.0)),
                    "rb_start": int(bm.rb_start),
                    "rb_width": int(bm.rb_width),
                })
        tel["beam_data"] = beam_data

        conflict_data = []
        outcome = self.env.last_resolution_outcome
        if outcome:
            for req in outcome.retained_requests + outcome.rejected_requests:
                is_ret = req in outcome.retained_requests
                conflict_data.append({
                    "sat_id": req.sat_id,
                    "shell_id": req.shell_id,
                    "beam_id": req.beam_id,
                    "cell_id": req.cell_id,
                    "is_retained": is_ret,
                    "power_share": float(req.power_share),
                    "rb_start": int(req.rb_start),
                    "rb_width": int(req.rb_width),
                    "channel_gain": float(req.channel_gain),
                    "propagation_delay_sec": float(req.propagation_delay_sec),
                    "queue_backlog_mb": float(req.queue_backlog_mb),
                    "resolution_score": float(outcome.resolution_scores.get((req.sat_id, req.beam_id), float("nan"))),
                    "rejection_reason": outcome.rejection_reasons.get((req.sat_id, req.beam_id), ""),
                })
        tel["conflict_data"] = conflict_data

        queues = self.env.traffic_model.get_queues()
        tel["cell_queues"] = queues.tolist()

        sat_snapshot = []
        for sat in self.env.satellite_network.satellites:
            lat, lon = _ecef_to_latlon(sat.position)
            sat_snapshot.append({
                "sat_id": sat.sat_id,
                "shell_id": sat.shell_id,
                "altitude_km": float(sat.altitude_km),
                "lat": lat,
                "lon": lon,
                "x_km": float(sat.position[0]),
                "y_km": float(sat.position[1]),
                "z_km": float(sat.position[2]),
                "true_anomaly_deg": float(sat.true_anomaly_deg),
                "visible_cells": list(sat.visible_cells),
                "num_visible": len(sat.visible_cells),
                "is_operational": getattr(sat, "is_operational", True),
            })
        tel["satellites"] = sat_snapshot

        cell_snapshot = []
        positions = self.env.traffic_model.get_positions()
        for c_idx, cell in enumerate(self.env.traffic_model.cells):
            clat, clon = _ecef_to_latlon(positions[c_idx])
            cell_snapshot.append({
                "cell_id": c_idx,
                "lat": clat,
                "lon": clon,
                "x_km": float(positions[c_idx][0]),
                "y_km": float(positions[c_idx][1]),
                "z_km": float(positions[c_idx][2]),
                "queue_mb": float(queues[c_idx]),
                "is_hotspot": cell.is_hotspot,
                "mean_arrival_rate": float(cell.mean_arrival_rate),
                "max_capacity": float(cell.max_capacity),
            })
        tel["cells"] = cell_snapshot

        return tel

    def get_satellite_snapshot(self) -> List[Dict[str, Any]]:
        """Return current satellite positions and state."""
        result = []
        for sat in self.env.satellite_network.satellites:
            lat, lon = _ecef_to_latlon(sat.position)
            result.append({
                "sat_id": sat.sat_id,
                "shell_id": sat.shell_id,
                "altitude_km": float(sat.altitude_km),
                "lat": lat,
                "lon": lon,
                "x_km": float(sat.position[0]),
                "y_km": float(sat.position[1]),
                "z_km": float(sat.position[2]),
                "visible_cells": list(sat.visible_cells),
                "is_operational": getattr(sat, "is_operational", True),
            })
        return result

    def get_cell_snapshot(self) -> List[Dict[str, Any]]:
        """Return current ground cell positions and queue states."""
        queues = self.env.traffic_model.get_queues()
        positions = self.env.traffic_model.get_positions()
        result = []
        for c_idx, cell in enumerate(self.env.traffic_model.cells):
            clat, clon = _ecef_to_latlon(positions[c_idx])
            result.append({
                "cell_id": c_idx,
                "lat": clat,
                "lon": clon,
                "x_km": float(positions[c_idx][0]),
                "y_km": float(positions[c_idx][1]),
                "z_km": float(positions[c_idx][2]),
                "queue_mb": float(queues[c_idx]),
                "is_hotspot": cell.is_hotspot,
                "mean_arrival_rate": float(cell.mean_arrival_rate),
                "max_capacity": float(cell.max_capacity),
                "cumulative_served": float(cell.cumulative_served),
                "cumulative_arrived": float(cell.cumulative_arrived),
            })
        return result

    def run_comparison(
        self,
        policy_names: List[str],
        seeds: List[int],
    ) -> Dict[str, Any]:
        """
        Run all specified policies on the same scenario and return comparison results.

        Args:
            policy_names: List of policy names to compare.
            seeds: Evaluation seeds.

        Returns:
            Dict mapping policy_name -> evaluation result dict.
        """
        results = {}
        for pname in policy_names:
            try:
                p = build_policy(pname, self.config, self.seed)
                res = evaluate_policy(self.env, p, pname, seeds)
                results[pname] = res
            except Exception as e:
                results[pname] = {"error": str(e)}
        return results

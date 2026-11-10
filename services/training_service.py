"""Safe background training worker and session coordinator for MAPPO."""
from __future__ import annotations
import os
import time
import threading
import copy
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from training.train import MAPPOTrainer
from training.checkpoint import save_checkpoint, load_checkpoint


class TrainingState:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class TrainingManager:
    """
    Manages MAPPO training execution in a background worker thread.
    Exposes thread-safe progress, telemetry, and control flags for Streamlit.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self.state: str = TrainingState.IDLE
        self.status_message: str = "Ready to train."
        self.current_iteration: int = 0
        self.total_iterations: int = 0
        self.total_slots: int = 0
        self.slots_trained: int = 0
        self.history: List[Dict[str, Any]] = []
        self.error_message: Optional[str] = None
        self.last_saved_checkpoint: Optional[str] = None
        self.trainer: Optional[MAPPOTrainer] = None

    def get_status(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of the current training state."""
        with self._lock:
            return {
                "state": self.state,
                "status_message": self.status_message,
                "current_iteration": self.current_iteration,
                "total_iterations": self.total_iterations,
                "slots_trained": self.slots_trained,
                "total_slots": self.total_slots,
                "progress_ratio": (self.slots_trained / max(1, self.total_slots)) if self.total_slots > 0 else 0.0,
                "history_length": len(self.history),
                "last_saved_checkpoint": self.last_saved_checkpoint,
                "error_message": self.error_message,
            }

    def get_history_df(self) -> pd.DataFrame:
        """Return training history as a pandas DataFrame."""
        with self._lock:
            if not self.history:
                return pd.DataFrame()
            return pd.DataFrame(self.history)

    def start_training(
        self,
        config: Dict[str, Any],
        total_slots: int = 500,
        eval_interval: int = 100,
        checkpoint_path: str = "checkpoints/best_model.pt",
        use_counterfactual_credit: bool = True,
    ) -> bool:
        """Launch background training thread."""
        with self._lock:
            if self.state == TrainingState.RUNNING:
                return False  # Already running

            self._stop_event.clear()
            self._pause_event.clear()
            self.state = TrainingState.RUNNING
            self.status_message = "Initializing trainer..."
            self.total_slots = total_slots
            self.slots_trained = 0
            self.current_iteration = 0
            self.history.clear()
            self.error_message = None

        def _run():
            try:
                # Deep copy config to ensure thread isolation
                cfg = copy.deepcopy(config)
                trainer = MAPPOTrainer(cfg)
                with self._lock:
                    self.trainer = trainer

                rollout_steps = trainer.rollout_steps
                iterations = max(1, total_slots // rollout_steps)
                with self._lock:
                    self.total_iterations = iterations

                current_obs, current_info = trainer.env.reset(seed=trainer.env.seed_val)
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

                iteration = 0
                while not self._stop_event.is_set() and iteration < iterations:
                    # Check pause
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        with self._lock:
                            self.state = TrainingState.PAUSED
                            self.status_message = "Training paused by user."
                        time.sleep(0.5)

                    if self._stop_event.is_set():
                        break

                    with self._lock:
                        self.state = TrainingState.RUNNING
                        self.status_message = f"Executing rollout iteration {iteration + 1}/{iterations}..."

                    t_start = time.perf_counter()
                    from training.rollout import collect_rollout
                    current_obs, current_info, rollout_tel = collect_rollout(
                        env=trainer.env,
                        agent=trainer.agent,
                        buffer=trainer.buffer,
                        use_counterfactual_credit=use_counterfactual_credit,
                        current_obs=current_obs,
                        current_info=current_info,
                    )

                    # Mini-batch PPO updates
                    from training.advantages import normalize_advantages
                    norm_adv = normalize_advantages(trainer.buffer.advantages)
                    total_samples = trainer.rollout_steps * trainer.num_satellites
                    batch_size = min(trainer.mini_batch_size, total_samples)

                    flat_obs = trainer.buffer.obs.reshape((total_samples, trainer.buffer.obs_dim))
                    flat_cell_acts = trainer.buffer.cell_actions.reshape((total_samples, trainer.beams_per_satellite))
                    flat_rb_acts = trainer.buffer.rb_actions.reshape((total_samples, trainer.beams_per_satellite))
                    flat_pwr_acts = trainer.buffer.power_actions.reshape((total_samples, trainer.beams_per_satellite))
                    flat_cell_masks = trainer.buffer.cell_masks.reshape((total_samples, trainer.buffer.num_cell_actions))
                    flat_rb_masks = trainer.buffer.rb_masks.reshape((total_samples, trainer.beams_per_satellite, trainer.buffer.num_rb_actions))
                    flat_old_lp = trainer.buffer.log_probs.reshape((total_samples,))
                    flat_adv = np.repeat(norm_adv, trainer.num_satellites, axis=0)
                    flat_globals = np.repeat(trainer.buffer.global_states, trainer.num_satellites, axis=0)
                    flat_returns = np.repeat(trainer.buffer.returns, trainer.num_satellites, axis=0).reshape((total_samples, 1))

                    flat_beam_cred = None
                    if use_counterfactual_credit:
                        flat_beam_cred = trainer.buffer.beam_credits.reshape((total_samples, trainer.beams_per_satellite))

                    epoch_losses = []
                    for _ in range(trainer.ppo_epochs):
                        indices = np.random.permutation(total_samples)
                        for start_idx in range(0, total_samples, batch_size):
                            batch_idx = indices[start_idx : start_idx + batch_size]
                            b_cred = flat_beam_cred[batch_idx] if flat_beam_cred is not None else None
                            loss_dict = trainer.agent.update(
                                batch_obs=flat_obs[batch_idx],
                                batch_cell_acts=flat_cell_acts[batch_idx],
                                batch_rb_acts=flat_rb_acts[batch_idx],
                                batch_pwr_acts=flat_pwr_acts[batch_idx],
                                batch_cell_masks=flat_cell_masks[batch_idx],
                                batch_rb_masks=flat_rb_masks[batch_idx],
                                batch_old_log_probs=flat_old_lp[batch_idx],
                                batch_advantages=flat_adv[batch_idx],
                                batch_global_states=flat_globals[batch_idx],
                                batch_returns=flat_returns[batch_idx],
                                batch_beam_credits=b_cred,
                            )
                            epoch_losses.append(loss_dict)

                    t_end = time.perf_counter()
                    mean_loss = {k: float(np.mean([l[k] for l in epoch_losses])) for k in epoch_losses[0].keys()}

                    iteration += 1
                    trained_slots = iteration * rollout_steps

                    # Periodic evaluation
                    eval_metrics = {}
                    if (trained_slots % eval_interval < rollout_steps) or (iteration >= iterations):
                        from evaluation.evaluate import run_episode
                        summary = run_episode(trainer.eval_env, trainer, seed=trainer.config.get("experiment", {}).get("seed", 42) + 999)
                        eval_metrics["eval_reward"] = summary.mean_team_reward
                        eval_metrics["eval_throughput"] = summary.mean_throughput_mbps
                        eval_metrics["eval_fairness"] = summary.mean_jains_fairness

                        if summary.mean_team_reward > trainer.best_eval_reward:
                            trainer.best_eval_reward = summary.mean_team_reward
                            save_checkpoint(
                                agent=trainer.agent,
                                filepath=checkpoint_path,
                                epoch=iteration,
                                metrics=eval_metrics,
                                config=trainer.config,
                            )
                            with self._lock:
                                self.last_saved_checkpoint = checkpoint_path

                    rec = {
                        "iteration": iteration,
                        "slots_trained": trained_slots,
                        "rollout_reward": float(rollout_tel.get("mean_reward", 0.0)),
                        "served_traffic_mb": float(rollout_tel.get("mean_served_mb", 0.0)),
                        "retention_rate": float(rollout_tel.get("mean_retention_rate", 0.0)),
                        "actor_loss": float(mean_loss.get("actor_loss", 0.0)),
                        "critic_loss": float(mean_loss.get("critic_loss", 0.0)),
                        "entropy": float(mean_loss.get("entropy", 0.0)),
                        "approx_kl": float(mean_loss.get("approx_kl", 0.0)),
                        "duration_sec": float(t_end - t_start),
                        "eval_reward": eval_metrics.get("eval_reward", np.nan),
                        "eval_throughput": eval_metrics.get("eval_throughput", np.nan),
                    }

                    with self._lock:
                        self.current_iteration = iteration
                        self.slots_trained = trained_slots
                        self.history.append(rec)

                with self._lock:
                    if self._stop_event.is_set():
                        self.state = TrainingState.STOPPED
                        self.status_message = "Training stopped by user."
                    else:
                        self.state = TrainingState.COMPLETED
                        self.status_message = f"Training completed successfully ({self.slots_trained} slots)."

            except Exception as e:
                import traceback
                err_detail = traceback.format_exc()
                with self._lock:
                    self.state = TrainingState.ERROR
                    self.error_message = f"{str(e)}\n{err_detail}"
                    self.status_message = f"Training failed: {str(e)}"

        self._worker_thread = threading.Thread(target=_run, daemon=True)
        self._worker_thread.start()
        return True

    def pause_training(self) -> None:
        """Signal pause to the worker thread."""
        self._pause_event.set()

    def resume_training(self) -> None:
        """Signal resume to the worker thread."""
        self._pause_event.clear()

    def stop_training(self) -> None:
        """Signal stop to the worker thread."""
        self._stop_event.set()
        self._pause_event.clear()

    def manual_save_checkpoint(self, path: str = "checkpoints/best_model.pt") -> bool:
        """Manually persist the active agent's weights."""
        with self._lock:
            if self.trainer is None:
                return False
            save_checkpoint(
                agent=self.trainer.agent,
                filepath=path,
                epoch=self.current_iteration,
                metrics={"slots_trained": self.slots_trained},
                config=self.trainer.config,
            )
            self.last_saved_checkpoint = path
            return True

    def manual_load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Manually load a checkpoint into the active agent."""
        with self._lock:
            if self.trainer is None:
                raise ValueError("Trainer not instantiated. Run or configure training first.")
            return load_checkpoint(self.trainer.agent, path)


# Global singleton instance for Streamlit process
_GLOBAL_TRAINING_MANAGER = TrainingManager()


def get_training_manager() -> TrainingManager:
    """Retrieve the shared singleton TrainingManager."""
    return _GLOBAL_TRAINING_MANAGER


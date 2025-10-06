"""
Callbacks personnalisés pour l'entraînement avec logging détaillé
"""

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import torch


class RichTBCallback(BaseCallback):
    """
    Callback TensorBoard enrichi qui logge:
    - Composantes de récompense détaillées (goal, touch, progress, etc.)
    - Taux de succès (goals marqués)
    - Norme des gradients pour détecter vanishing/exploding
    - Statistiques d'épisodes (returns, lengths)
    - Histogrammes des récompenses
    """

    def __init__(self, log_dir, reward_keys=("goal", "touch", "progress", "demo"), verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.reward_keys = reward_keys
        self._last = time.time()
        self._step = 0
        self._ep_returns = []
        self._ep_lens = []
        self._reward_parts = {k: [] for k in reward_keys}
        self._success = []

    def _on_step(self) -> bool:
        self._step += 1
        infos = self.locals.get("infos", [])

        for info in infos:
            # Parse les composantes de récompense dans info["reward_parts"]
            parts = info.get("reward_parts")
            if parts:
                for k in self.reward_keys:
                    if k in parts:
                        self._reward_parts[k].append(parts[k])

            # Taux de succès
            if "success" in info:
                self._success.append(float(info["success"]))

            # Statistiques d'épisode
            if "episode" in info:
                self._ep_returns.append(info["episode"]["r"])
                self._ep_lens.append(info["episode"]["l"])

        # Log toutes les N steps
        if self._step % 1000 == 0:
            t = self.num_timesteps

            # Statistiques d'épisodes
            if self._ep_returns:
                self.writer.add_scalar("rollout/ep_rew_mean", np.mean(self._ep_returns), t)
                self.writer.add_scalar("rollout/ep_len_mean", np.mean(self._ep_lens), t)
                self._ep_returns.clear()
                self._ep_lens.clear()

            # Taux de succès
            if self._success:
                self.writer.add_scalar("metrics/success_rate", np.mean(self._success), t)
                self._success.clear()

            # Composantes de récompense (moyennes + histogrammes)
            for k, vals in self._reward_parts.items():
                if vals:
                    self.writer.add_scalar(f"reward/{k}_mean", np.mean(vals), t)
                    self.writer.add_histogram(f"reward/{k}_hist", np.array(vals), t)
                    self._reward_parts[k].clear()

            # Norme des gradients (si PPO SB3)
            model = self.model
            if model and hasattr(model, "policy"):
                total_norm = 0.0
                for p in model.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("train/grad_norm", total_norm, t)

        return True

    def _on_training_end(self) -> None:
        self.writer.flush()
        self.writer.close()

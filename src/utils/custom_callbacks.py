"""
Callbacks personnalisés pour notre implémentation PPO custom (non-SB3)
"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import torch


class CustomTBCallback:
    """
    Callback TensorBoard pour notre implémentation custom de PPO.
    Logge les composantes de récompense, statistiques d'épisodes, gradients, etc.
    """

    def __init__(self, log_dir, reward_keys=("goal", "touch", "progress", "boost", "demo", "aerial"), verbose=0):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.reward_keys = reward_keys
        self.verbose = verbose
        self._step = 0
        self._ep_returns = []
        self._ep_lens = []
        self._reward_parts = {k: [] for k in reward_keys}
        self._success = []
        self._last_time = time.time()

    def on_step(self, info, timestep):
        """
        Appelé après chaque step de l'environnement.

        Args:
            info: Dictionary contenant les informations du step (peut inclure reward_parts, success, episode)
            timestep: Numéro du timestep actuel
        """
        self._step += 1

        # Parse les composantes de récompense dans info["reward_parts"]
        if "reward_parts" in info:
            parts = info["reward_parts"]
            for k in self.reward_keys:
                if k in parts:
                    self._reward_parts[k].append(parts[k])

        # Taux de succès (défini quand un but est marqué)
        if "success" in info:
            self._success.append(float(info["success"]))

        # Statistiques d'épisode
        if "episode" in info:
            self._ep_returns.append(info["episode"]["r"])
            self._ep_lens.append(info["episode"]["l"])

        # Log toutes les N steps
        if self._step % 1000 == 0:
            self._log_stats(timestep)

    def on_train_step(self, train_stats, timestep, agent=None):
        """
        Appelé après chaque étape d'entraînement.

        Args:
            train_stats: Dictionary contenant les stats d'entraînement (policy_loss, value_loss, entropy)
            timestep: Numéro du timestep actuel
            agent: Agent PPO (pour logger les gradients)
        """
        # Logger les stats d'entraînement
        for key, value in train_stats.items():
            self.writer.add_scalar(f"train/{key}", value, timestep)

        # Logger la norme des gradients SEULEMENT tous les 10 updates (éviter overhead)
        if agent is not None and hasattr(agent, "policy_network") and timestep % 10000 == 0:
            total_norm = 0.0
            for p in agent.policy_network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar("train/grad_norm", total_norm, timestep)

    def _log_stats(self, timestep):
        """Log les statistiques accumulées"""
        # Statistiques d'épisodes
        if self._ep_returns:
            self.writer.add_scalar("rollout/ep_rew_mean", np.mean(self._ep_returns), timestep)
            self.writer.add_scalar("rollout/ep_rew_std", np.std(self._ep_returns), timestep)
            if self.verbose:
                print(f"[Step {self._step}] Avg episode reward: {np.mean(self._ep_returns):.2f}")
            self._ep_returns.clear()

        if self._ep_lens:
            self.writer.add_scalar("rollout/ep_len_mean", np.mean(self._ep_lens), timestep)
            self._ep_lens.clear()

        # Taux de succès
        if self._success:
            success_rate = np.mean(self._success)
            self.writer.add_scalar("metrics/success_rate", success_rate, timestep)
            if self.verbose:
                print(f"[Step {self._step}] Success rate: {success_rate:.2%}")
            self._success.clear()

        # Composantes de récompense (moyennes + histogrammes)
        for k, vals in self._reward_parts.items():
            if vals:
                self.writer.add_scalar(f"reward/{k}_mean", np.mean(vals), timestep)
                self.writer.add_histogram(f"reward/{k}_hist", np.array(vals), timestep)
                self._reward_parts[k].clear()

        # FPS
        current_time = time.time()
        elapsed = current_time - self._last_time
        if elapsed > 0:
            fps = 1000 / elapsed
            self.writer.add_scalar("perf/fps", fps, timestep)
        self._last_time = current_time

    def on_checkpoint_save(self, checkpoint_path, timestep, reward):
        """
        Appelé quand un checkpoint est sauvegardé.

        Args:
            checkpoint_path: Chemin du checkpoint
            timestep: Numéro du timestep
            reward: Récompense actuelle
        """
        if self.verbose:
            print(f"Checkpoint saved at {checkpoint_path} (timestep={timestep}, reward={reward:.2f})")

    def close(self):
        """Ferme le writer"""
        self.writer.flush()
        self.writer.close()


class EvalCallback:
    """
    Callback pour évaluation périodique et sauvegarde du meilleur modèle.
    """

    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=10,
                 best_model_path="./checkpoints/best_model.pth", verbose=0):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.verbose = verbose
        self.best_mean_reward = -float('inf')
        self.last_eval_timestep = 0

        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    def should_eval(self, timestep):
        """Vérifie si on doit faire une évaluation"""
        return (timestep - self.last_eval_timestep) >= self.eval_freq

    def evaluate(self, agent, timestep, writer=None):
        """
        Évalue l'agent sur n_eval_episodes.

        Args:
            agent: Agent PPO à évaluer
            timestep: Numéro du timestep actuel
            writer: TensorBoard writer (optionnel)

        Returns:
            mean_reward: Récompense moyenne sur les épisodes d'évaluation
        """
        if not self.should_eval(timestep):
            return None

        self.last_eval_timestep = timestep
        episode_rewards = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                # Sélection d'action déterministe (sans exploration)
                with torch.no_grad():
                    action, _, _ = agent.select_action(obs, deterministic=True)

                obs, reward, terminated, truncated, _ = self.eval_env.step(int(action))
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        # Logger
        if writer:
            writer.add_scalar("eval/mean_reward", mean_reward, timestep)
            writer.add_scalar("eval/std_reward", std_reward, timestep)
            writer.add_scalar("eval/mean_length", mean_length, timestep)

        if self.verbose:
            print(f"\n[Eval @ {timestep}] Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Sauvegarder si c'est le meilleur
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            agent.save(self.best_model_path)
            if self.verbose:
                print(f"New best model saved! Reward: {mean_reward:.2f}")

        return mean_reward

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List


class RolloutBuffer:
    """
    Buffer pour stocker les trajectoires d'expérience pour PPO
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device='cpu'):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Buffers
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32).to(device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32).to(device)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, log_prob, reward, value, done):
        """Ajoute une transition au buffer"""
        self.observations[self.ptr] = torch.FloatTensor(obs).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Calcule les returns et advantages avec GAE (Generalized Advantage Estimation)
        """
        advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lambda = 0

        size = self.buffer_size if self.full else self.ptr

        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda

        self.advantages[:size] = advantages[:size]
        self.returns[:size] = advantages[:size] + self.values[:size]

        # Normaliser les advantages
        self.advantages[:size] = (self.advantages[:size] - self.advantages[:size].mean()) / (
            self.advantages[:size].std() + 1e-8
        )

    def get(self):
        """Récupère toutes les données du buffer"""
        size = self.buffer_size if self.full else self.ptr

        return {
            'observations': self.observations[:size],
            'actions': self.actions[:size],
            'log_probs': self.log_probs[:size],
            'advantages': self.advantages[:size],
            'returns': self.returns[:size],
            'values': self.values[:size]
        }

    def clear(self):
        """Vide le buffer"""
        self.ptr = 0
        self.full = False


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm
    """

    def __init__(
        self,
        policy_network,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        target_kl=None,
        device='cpu'
    ):
        self.policy = policy_network.to(device)
        self.device = device

        # Hyperparamètres
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Statistiques
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }

    def select_action(self, obs, deterministic=False):
        """
        Sélectionne une action basée sur l'observation

        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)

            return (
                action.cpu().numpy()[0],
                log_prob.cpu().item(),
                value.cpu().item()
            )

    def train(self, rollout_buffer: RolloutBuffer, n_epochs=10, batch_size=64):
        """
        Entraîne le policy network avec les données du rollout buffer

        Args:
            rollout_buffer: Buffer contenant les expériences
            n_epochs: Nombre d'époques d'entraînement
            batch_size: Taille des mini-batches
        """
        # Récupérer les données
        data = rollout_buffer.get()
        dataset_size = data['observations'].shape[0]

        # Reset des statistiques
        for key in self.train_stats:
            self.train_stats[key] = []

        # Training loop
        for epoch in range(n_epochs):
            # Shuffle des indices
            indices = torch.randperm(dataset_size)

            # Mini-batch training
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                # Extraire le mini-batch
                obs_batch = data['observations'][batch_indices]
                actions_batch = data['actions'][batch_indices]
                old_log_probs_batch = data['log_probs'][batch_indices]
                advantages_batch = data['advantages'][batch_indices]
                returns_batch = data['returns'][batch_indices]
                old_values_batch = data['values'][batch_indices]

                # Évaluer les actions avec le policy actuel
                log_probs, entropy, values = self.policy.evaluate_actions(obs_batch, actions_batch)
                values = values.squeeze()

                # === Policy Loss (PPO clip) ===
                # Ratio entre nouvelle et ancienne policy
                ratio = torch.exp(log_probs - old_log_probs_batch)

                # Surrogate loss
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # === Value Loss ===
                # Clipping de la value function (optionnel)
                values_clipped = old_values_batch + torch.clamp(
                    values - old_values_batch,
                    -self.clip_range,
                    self.clip_range
                )

                value_loss_unclipped = (values - returns_batch).pow(2)
                value_loss_clipped = (values_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # === Entropy Loss ===
                entropy_loss = -entropy.mean()

                # === Total Loss ===
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # === Statistiques ===
                with torch.no_grad():
                    # KL divergence approximation
                    kl = (old_log_probs_batch - log_probs).mean()

                    # Fraction d'actions clippées
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                    # Enregistrer les stats
                    self.train_stats['policy_loss'].append(policy_loss.item())
                    self.train_stats['value_loss'].append(value_loss.item())
                    self.train_stats['entropy_loss'].append(entropy_loss.item())
                    self.train_stats['total_loss'].append(loss.item())
                    self.train_stats['kl_divergence'].append(kl.item())
                    self.train_stats['clip_fraction'].append(clip_fraction.item())

            # Early stopping si KL divergence trop élevée
            if self.target_kl is not None:
                mean_kl = np.mean(self.train_stats['kl_divergence'][-10:])
                if mean_kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to high KL divergence: {mean_kl:.4f}")
                    break

        # Moyennes des statistiques
        return {
            key: np.mean(values) for key, values in self.train_stats.items()
        }

    def save(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

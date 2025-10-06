"""
Wrappers vectorisés et utilitaires pour multi-environnements
"""

import numpy as np
import pickle
import os


class RewardPartsToInfo:
    """
    Wrapper minimal (10 lignes) pour copier reward_parts dans info.
    Résout le problème de propagation shared_info → info.
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Copier reward_parts depuis shared_info si disponible
        if hasattr(self.env, 'reward_fn') and hasattr(self.env.reward_fn, '_last_reward_parts'):
            info['reward_parts'] = self.env.reward_fn._last_reward_parts

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class SimpleVecNormalize:
    """
    Normalisation simple des observations et returns (sans SB3).
    Maintient des stats running mean/std et sauvegarde dans un .pkl.
    """

    def __init__(self, normalize_obs=True, normalize_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8):
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        # Running statistics
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = epsilon

        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = epsilon

        # Return tracking
        self.returns = None

    def normalize_obs_array(self, obs):
        """Normalise les observations"""
        if not self._normalize_obs:
            return obs

        if self.obs_mean is None:
            self.obs_mean = np.zeros_like(obs, dtype=np.float64)
            self.obs_var = np.ones_like(obs, dtype=np.float64)

        # Update running stats
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count

        self.obs_mean = self.obs_mean + delta * batch_count / tot_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / tot_count
        self.obs_var = M2 / tot_count
        self.obs_count = tot_count

        # Normalize
        obs_norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(obs_norm, -self.clip_obs, self.clip_obs)

    def normalize_reward(self, reward, done):
        """Normalise les récompenses"""
        if not self._normalize_reward:
            return reward

        if self.returns is None:
            self.returns = np.zeros_like(reward, dtype=np.float64)

        # Update returns
        self.returns = self.returns * self.gamma + reward

        # Update running stats
        batch_mean = np.mean(self.returns)
        batch_var = np.var(self.returns)
        batch_count = len(reward)

        delta = batch_mean - self.ret_mean
        tot_count = self.ret_count + batch_count

        self.ret_mean = self.ret_mean + delta * batch_count / tot_count
        M2 = self.ret_var * self.ret_count + batch_var * batch_count + \
             np.square(delta) * self.ret_count * batch_count / tot_count
        self.ret_var = M2 / tot_count
        self.ret_count = tot_count

        # Reset returns for done episodes
        self.returns[done] = 0.0

        # Normalize
        reward_norm = reward / np.sqrt(self.ret_var + self.epsilon)
        return np.clip(reward_norm, -self.clip_reward, self.clip_reward)

    def save(self, path):
        """Sauvegarde les stats de normalisation"""
        stats = {
            'obs_mean': self.obs_mean,
            'obs_var': self.obs_var,
            'obs_count': self.obs_count,
            'ret_mean': self.ret_mean,
            'ret_var': self.ret_var,
            'ret_count': self.ret_count,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(stats, f)

    def load(self, path):
        """Charge les stats de normalisation"""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        self.obs_mean = stats['obs_mean']
        self.obs_var = stats['obs_var']
        self.obs_count = stats['obs_count']
        self.ret_mean = stats['ret_mean']
        self.ret_var = stats['ret_var']
        self.ret_count = stats['ret_count']


class LRScheduler:
    """
    Learning rate scheduler simple (linéaire ou cosine).
    """

    def __init__(self, optimizer, initial_lr, final_lr, total_steps, schedule_type='linear'):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.schedule_type == 'linear':
            progress = min(1.0, self.current_step / self.total_steps)
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        elif self.schedule_type == 'cosine':
            progress = min(1.0, self.current_step / self.total_steps)
            lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * \
                 (1 + np.cos(np.pi * progress))
        else:
            lr = self.initial_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        """Retourne le learning rate actuel"""
        return self.optimizer.param_groups[0]['lr']

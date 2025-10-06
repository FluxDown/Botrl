"""
Wrapper pour environnement RLGym qui propage reward_parts et détecte les succès
"""

import numpy as np
from rlgym.api import RLGym


class RewardPartsWrapper:
    """
    Wrapper qui propage les reward_parts depuis shared_info vers info
    et détecte automatiquement les succès (buts marqués)
    """

    def __init__(self, env: RLGym):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_blue_score = 0
        self._last_orange_score = 0

    def reset(self, **kwargs):
        """Reset l'environnement"""
        result = self.env.reset(**kwargs)

        # RLGym 2.0 peut retourner (obs, info) ou juste obs
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Initialiser les scores
        self._last_blue_score = 0
        self._last_orange_score = 0

        return obs, info

    def step(self, action):
        """
        Step dans l'environnement avec propagation de reward_parts
        """
        # Appeler l'env de base
        result = self.env.step(action)

        # Gérer les retours (4 ou 5 valeurs selon version)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, truncated = result
            info = {}

        # Récupérer shared_info si disponible
        if hasattr(self.env, '_shared_info') and self.env._shared_info:
            shared_info = self.env._shared_info

            # Propager reward_parts dans info
            if 'reward_parts' in shared_info:
                # Pour un seul agent
                if isinstance(obs, np.ndarray):
                    info['reward_parts'] = shared_info['reward_parts'].get(0, {})
                # Pour plusieurs agents (dict)
                elif isinstance(obs, dict):
                    # Prendre les reward_parts du premier agent (ou faire une moyenne)
                    agent_ids = list(obs.keys())
                    if agent_ids:
                        first_agent = agent_ids[0]
                        info['reward_parts'] = shared_info['reward_parts'].get(first_agent, {})

        # Détecter les succès (but marqué)
        # Note: cela nécessite d'accéder au GameState
        info['success'] = 0.0

        if hasattr(self.env, '_game_state') and self.env._game_state:
            current_blue_score = self.env._game_state.blue_score
            current_orange_score = self.env._game_state.orange_score

            # Un but a été marqué si le score a changé
            if current_blue_score > self._last_blue_score or current_orange_score > self._last_orange_score:
                info['success'] = 1.0

            self._last_blue_score = current_blue_score
            self._last_orange_score = current_orange_score

        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        """Render l'environnement"""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Ferme l'environnement"""
        return self.env.close()

    def __getattr__(self, name):
        """Délègue les attributs non trouvés à l'env de base"""
        return getattr(self.env, name)


class RLGymInfoWrapper:
    """
    Wrapper alternatif qui injecte reward_parts directement dans le flow
    Compatible avec RLGym 2.0.1
    """

    def __init__(self, env: RLGym, reward_fn_ref=None):
        """
        Args:
            env: Environnement RLGym
            reward_fn_ref: Référence à la reward function pour accéder aux reward_parts
        """
        self.env = env
        self.reward_fn = reward_fn_ref
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._episode_reward_parts = {
            "goal": [],
            "touch": [],
            "progress": [],
            "boost": [],
            "demo": [],
            "aerial": []
        }
        self._last_blue_score = 0
        self._last_orange_score = 0

    def reset(self, **kwargs):
        """Reset"""
        result = self.env.reset(**kwargs)

        # Clear reward parts tracking
        for key in self._episode_reward_parts:
            self._episode_reward_parts[key] = []

        self._last_blue_score = 0
        self._last_orange_score = 0

        if isinstance(result, tuple):
            return result
        return result, {}

    def step(self, action):
        """Step avec injection de reward_parts"""
        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, truncated = result
            info = {}

        # Injecter les reward_parts accumulées
        if self._episode_reward_parts:
            # Calculer les moyennes pour ce step
            reward_parts_avg = {}
            for key, values in self._episode_reward_parts.items():
                if values:
                    reward_parts_avg[key] = np.mean(values[-10:])  # Moyenne sur les 10 derniers
                else:
                    reward_parts_avg[key] = 0.0

            info['reward_parts'] = reward_parts_avg

        # Détecter success (simplifié - basé sur la récompense)
        info['success'] = 1.0 if reward > 50 else 0.0

        return obs, reward, terminated, truncated, info

    def add_reward_parts(self, reward_parts: dict):
        """
        Méthode publique pour ajouter les reward_parts depuis l'extérieur
        À appeler depuis votre boucle d'entraînement si vous avez accès aux reward_parts
        """
        for key, value in reward_parts.items():
            if key in self._episode_reward_parts:
                self._episode_reward_parts[key].append(value)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

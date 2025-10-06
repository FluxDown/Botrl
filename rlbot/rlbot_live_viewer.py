"""
RLBot Live Viewer - Visualise ton bot pendant l'entraînement

Usage:
1. Lance train_mp.py (entraînement en arrière-plan)
2. Lance ce script avec RLBot
3. Ouvre Rocket League et regarde ton bot jouer en temps réel

Le bot recharge automatiquement les poids toutes les 1-2 secondes.
"""

import os
import time
import torch
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# Importe ton réseau (pas d'autres dépendances pour éviter les conflits)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn


# ========== ActorCritic Network (copie standalone) ==========
class ActorCritic(nn.Module):
    """Version simplifiée standalone du réseau"""

    def __init__(self, obs_space_size, action_space_size, policy_layers=[256, 256, 256],
                 value_layers=[256, 256, 256], activation='relu', continuous_actions=False):
        super().__init__()
        self.continuous_actions = continuous_actions

        # Activation
        act_fn = nn.ReLU if activation == 'relu' else nn.Tanh

        # Policy network
        policy_net = []
        in_size = obs_space_size
        for hidden_size in policy_layers:
            policy_net.append(nn.Linear(in_size, hidden_size))
            policy_net.append(act_fn())
            in_size = hidden_size
        self.policy_net = nn.Sequential(*policy_net)
        self.policy_head = nn.Linear(in_size, action_space_size)

        # Value network
        value_net = []
        in_size = obs_space_size
        for hidden_size in value_layers:
            value_net.append(nn.Linear(in_size, hidden_size))
            value_net.append(act_fn())
            in_size = hidden_size
        self.value_net = nn.Sequential(*value_net)
        self.value_head = nn.Linear(in_size, 1)

    def forward(self, obs):
        policy_features = self.policy_net(obs)
        action_logits = self.policy_head(policy_features)

        value_features = self.value_net(obs)
        value = self.value_head(value_features)

        return action_logits, value

    def get_action(self, obs, deterministic=False):
        action_logits, value = self(obs)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = torch.log_softmax(action_logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob, value.squeeze(-1)


# ========== Lookup Table Actions (90 actions) ==========
LOOKUP_TABLE = [
    # Format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    # ... (tu peux ajouter les 90 actions complètes ou utiliser une version simplifiée)
    # Pour l'instant, on génère aléatoirement
]

# Générer une lookup table simple
for i in range(90):
    LOOKUP_TABLE.append([
        np.random.uniform(-1, 1),  # throttle
        np.random.uniform(-1, 1),  # steer
        np.random.uniform(-1, 1),  # pitch
        np.random.uniform(-1, 1),  # yaw
        np.random.uniform(-1, 1),  # roll
        np.random.randint(0, 2),   # jump
        np.random.randint(0, 2),   # boost
        np.random.randint(0, 2),   # handbrake
    ])


class LivePolicy:
    """Charge et recharge automatiquement le policy depuis latest_policy.pt"""

    def __init__(self, policy_path, obs_dim=None, act_dim=90):
        self.policy_path = policy_path
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.last_mtime = 0
        self.last_reload = 0
        self.reload_interval = 2.0  # Recharger toutes les 2 secondes

        # Stats de normalisation
        self.obs_mean = None
        self.obs_var = None

        # Créer le réseau
        if obs_dim is None:
            # Attendre le premier reload pour connaître obs_dim
            self.policy = None
        else:
            self.policy = ActorCritic(
                obs_space_size=obs_dim,
                action_space_size=act_dim,
                policy_layers=[256, 256, 256],
                value_layers=[256, 256, 256],
                activation='relu',
                continuous_actions=False
            )
            self.policy.eval()

        # Tenter un premier chargement
        self.maybe_reload(force=True)

    def maybe_reload(self, force=False):
        """Recharge le policy si le fichier a changé"""
        now = time.time()

        # Throttle: ne pas recharger trop souvent
        if not force and (now - self.last_reload) < self.reload_interval:
            return False

        try:
            # Vérifier si le fichier existe et a changé
            if not os.path.exists(self.policy_path):
                return False

            mtime = os.path.getmtime(self.policy_path)
            if not force and mtime <= self.last_mtime:
                return False

            # Charger les poids
            checkpoint = torch.load(self.policy_path, map_location='cpu')

            # Première fois: créer le réseau si besoin
            if self.policy is None:
                # Deviner obs_dim depuis les poids
                first_layer = checkpoint['policy_state_dict']['policy_net.0.weight']
                obs_dim = first_layer.shape[1]

                self.policy = ActorCritic(
                    obs_space_size=obs_dim,
                    action_space_size=self.act_dim,
                    policy_layers=[256, 256, 256],
                    value_layers=[256, 256, 256],
                    activation='relu',
                    continuous_actions=False
                )
                print(f"[LivePolicy] Created network with obs_dim={obs_dim}")

            # Charger les poids
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.eval()

            # Charger les stats de normalisation
            self.obs_mean = checkpoint.get('obs_mean')
            self.obs_var = checkpoint.get('obs_var')

            self.last_mtime = mtime
            self.last_reload = now

            total_steps = checkpoint.get('total_steps', '?')
            print(f"[LivePolicy] Reloaded weights @ {total_steps} steps")
            return True

        except Exception as e:
            print(f"[LivePolicy] Failed to reload: {e}")
            return False

    @torch.no_grad()
    def act(self, obs_vec, deterministic=True):
        """Prédit une action depuis l'observation"""
        if self.policy is None:
            return 0  # Noop si pas encore chargé

        # Normaliser l'observation
        if self.obs_mean is not None and self.obs_var is not None:
            obs_normalized = (obs_vec - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            obs_normalized = np.clip(obs_normalized, -10, 10)
        else:
            obs_normalized = obs_vec

        # Forward
        obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
        action_logits, _ = self.policy.get_action(obs_tensor, deterministic=deterministic)

        return int(action_logits[0].item())


class RLBotLiveViewer(BaseAgent):
    """
    Bot RLBot qui visualise ton agent en temps réel pendant l'entraînement
    """

    def initialize_agent(self):
        # Chemin vers latest_policy.pt
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        policy_path = os.path.join(checkpoint_dir, 'latest_policy.pt')

        # Charger le policy
        self.policy = LivePolicy(policy_path, obs_dim=None, act_dim=90)

        print(f"[RLBotLiveViewer] Initialized! Watching {policy_path}")
        print(f"[RLBotLiveViewer] Policy will auto-reload every 2 seconds")
        print(f"[RLBotLiveViewer] WARNING: Using placeholder observations - bot will not play correctly yet!")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Appelé à chaque tick (~60 Hz)"""

        # Recharger le policy si nécessaire (throttled)
        self.policy.maybe_reload()

        # TODO: Construire l'observation depuis le GameTickPacket
        # Pour l'instant, utilise des zeros comme placeholder
        # L'obs_dim sera détecté automatiquement depuis les poids du modèle

        # Placeholder: observation vide (sera adapté automatiquement à la bonne taille)
        # En attendant l'implémentation complète de l'observation
        if self.policy.policy is not None:
            obs_dim = self.policy.policy.policy_net[0].in_features
            obs = np.zeros(obs_dim, dtype=np.float32)
        else:
            obs = np.zeros(107, dtype=np.float32)  # Taille par défaut

        action_idx = self.policy.act(obs)

        # Mapper vers ControllerState via LOOKUP_TABLE
        if action_idx < len(LOOKUP_TABLE):
            action_array = LOOKUP_TABLE[action_idx]
        else:
            action_array = [0, 0, 0, 0, 0, 0, 0, 0]  # Noop

        ctrl = SimpleControllerState()
        ctrl.throttle = float(action_array[0])
        ctrl.steer = float(action_array[1])
        ctrl.pitch = float(action_array[2])
        ctrl.yaw = float(action_array[3])
        ctrl.roll = float(action_array[4])
        ctrl.jump = bool(action_array[5])
        ctrl.boost = bool(action_array[6])
        ctrl.handbrake = bool(action_array[7])

        return ctrl

# rlbot/policy_loader.py
import os
import time
import torch
import numpy as np
import torch.nn as nn

# --- construit le modèle exact de l'entraînement ---
def build_model(obs_dim: int, n_actions: int):
    """
    Architecture EXACTE de src/networks/actor_critic.py
    3 layers de 256 neurons + ReLU
    """
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim, n_actions):
            super().__init__()
            # Policy: 3 layers
            self.policy_net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.policy_head = nn.Linear(256, n_actions)

            # Value: 3 layers
            self.value_net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.value_head = nn.Linear(256, 1)

        def forward(self, x):
            policy_features = self.policy_net(x)
            logits = self.policy_head(policy_features)
            value_features = self.value_net(x)
            value = self.value_head(value_features)
            return logits, value

    return ActorCritic(obs_dim, n_actions)

class LivePolicy:
    """
    Recharge latest_policy.pt quand il change.
    """
    def __init__(self, weights_path: str, obs_dim: int, n_actions: int, device: str = "cpu"):
        self.path = weights_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_model(obs_dim, n_actions).to(self.device).eval()
        self.last_mtime = 0.0
        self.warm = False

    def maybe_reload(self):
        try:
            m = os.path.getmtime(self.path)
            if m > self.last_mtime:
                sd = torch.load(self.path, map_location=self.device)
                self.model.load_state_dict(sd, strict=False)
                self.model.eval()
                self.last_mtime = m
        except FileNotFoundError:
            pass
        except Exception as e:
            # on ignore les erreurs de lecture ponctuelles pendant l'écriture
            # (assure un os.replace côté trainer)
            print(f"[LivePolicy] reload error: {e}")

    @torch.no_grad()
    def act_discrete(self, obs_vec: np.ndarray) -> int:
        self.maybe_reload()
        x = torch.from_numpy(obs_vec).float().to(self.device).unsqueeze(0)  # (1, obs_dim)
        logits, _ = self.model(x)
        a = torch.argmax(logits, dim=-1).item()
        return int(a)

import os, torch, numpy as np

def build_model(obs_dim: int, n_actions: int):
    import torch.nn as nn
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim, n_actions):
            super().__init__()
            self.pi = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, n_actions),
            )
            self.v = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )
        def forward(self, x):
            return self.pi(x), self.v(x)
    return ActorCritic(obs_dim, n_actions)

class LivePolicy:
    def __init__(self, weights_path: str, obs_dim: int, n_actions: int, device="cpu"):
        self.path = weights_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_model(obs_dim, n_actions).to(self.device).eval()
        self.last_mtime = 0.0
        self.n_actions = n_actions

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
            print(f"[LivePolicy] reload error: {e}")

    @torch.no_grad()
    def act_discrete(self, obs_vec: np.ndarray) -> int:
        self.maybe_reload()
        x = torch.from_numpy(obs_vec).float().to(self.device).unsqueeze(0)
        try:
            logits, _ = self.model(x)  # (1, n_actions)
            a = int(torch.argmax(logits, dim=-1).item())
        except Exception:
            # Si pas de poids valides encore: action random
            a = int(np.random.randint(self.n_actions))
        return a

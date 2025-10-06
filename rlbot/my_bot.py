# rlbot/my_bot.py
import os
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import Vector3

from .policy_loader import LivePolicy
from .actions import map_action_index

# ------- OBS BUILDER -------
# ⚠️ TODO: remplace par la même logique que ton AdvancedObs d'entraînement !
def build_obs_from_packet(packet, index: int) -> np.ndarray:
    # Ex: features rapides [pos(x,y,z), vel(x,y,z), ball pos, ball vel, has_boost, on_ground]
    me = packet.game_cars[index]
    ball = packet.game_ball
    pos = me.physics.location
    vel = me.physics.velocity
    bpos = ball.physics.location
    bvel = ball.physics.velocity

    has_boost = float(me.boost > 0)
    on_ground = float(me.has_wheel_contact)

    # Normalisations très grossières (adapte aux bornes de ton AdvancedObs)
    obs = np.array([
        pos.x/4096, pos.y/5120, pos.z/2044,
        vel.x/2300, vel.y/2300, vel.z/2300,
        bpos.x/4096, bpos.y/5120, bpos.z/2044,
        bvel.x/6000, bvel.y/6000, bvel.z/6000,
        has_boost, on_ground
    ], dtype=np.float32)
    return obs

class MyLiveBot(BaseAgent):
    def initialize_agent(self):
        # Définis ton obs_dim et n_actions (doivent matcher le training !)
        self.obs_dim = 45  # AdvancedObs pour team_size=1
        self.n_actions = 90  # LookupTableAction
        weights_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "latest_policy.pt")
        weights_path = os.path.abspath(weights_path)

        # CPU par défaut (RLBot tourne hors CUDA souvent)
        self.policy = LivePolicy(weights_path, obs_dim=self.obs_dim, n_actions=self.n_actions, device="cpu")

    def get_output(self, packet) -> SimpleControllerState:
        obs_vec = build_obs_from_packet(packet, self.index)
        # Optionnel: pad/troncature si obs_dim != len(obs_vec)
        if obs_vec.shape[0] != self.obs_dim:
            if obs_vec.shape[0] > self.obs_dim:
                obs_vec = obs_vec[:self.obs_dim]
            else:
                obs_vec = np.pad(obs_vec, (0, self.obs_dim - obs_vec.shape[0]), mode="constant")

        a_idx = self.policy.act_discrete(obs_vec)
        ctrl = map_action_index(a_idx)
        return ctrl

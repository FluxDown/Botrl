import os, numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from .policy_loader import LivePolicy
from .actions import map_action_index, n_actions

# ⚠️ TODO: remplace par la même obs que ton AdvancedObs de training
def build_obs_from_packet(packet, index: int) -> np.ndarray:
    me = packet.game_cars[index]
    ball = packet.game_ball
    pos = me.physics.location; vel = me.physics.velocity
    bpos = ball.physics.location; bvel = ball.physics.velocity
    has_boost = float(me.boost > 0); on_ground = float(me.has_wheel_contact)
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
        self.obs_dim = 14                  # ⚠️ aligne avec AdvancedObs réel
        self.n_act   = n_actions()         # 27 ici; mets 90 si tu charges ta LUT
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "latest_policy.pt"))
        self.policy = LivePolicy(weights_path, obs_dim=self.obs_dim, n_actions=self.n_act, device="cpu")

    def get_output(self, packet) -> SimpleControllerState:
        obs = build_obs_from_packet(packet, self.index)
        # pad/tronc si besoin
        if obs.shape[0] != self.obs_dim:
            if obs.shape[0] > self.obs_dim: obs = obs[:self.obs_dim]
            else: obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]), mode="constant")
        a_idx = self.policy.act_discrete(obs)
        return map_action_index(a_idx)

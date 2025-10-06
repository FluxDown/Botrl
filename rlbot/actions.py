import numpy as np
from rlbot.agents.base_agent import SimpleControllerState

# Mini-LUT simple (27 actions) — remplace par ta LUT 90 actions si tu l'exportes depuis ton training
_LUT = []
for throttle in [0.0, 1.0]:
    for steer in [-1.0, 0.0, 1.0]:
        for boost in [0.0, 1.0]:
            for jump in [0.0, 1.0]:
                _LUT.append([throttle, steer, 0.0, steer, 0.0, jump, boost, 0.0])
_LUT = np.array(_LUT, dtype=np.float32)

def map_action_index(a_idx: int) -> SimpleControllerState:
    a_idx = int(a_idx) % len(_LUT)
    thr, steer, pitch, yaw, roll, jump, boost, handbrake = _LUT[a_idx]
    ctrl = SimpleControllerState()
    ctrl.throttle  = float(thr)
    ctrl.steer     = float(steer)
    ctrl.pitch     = float(pitch)
    ctrl.yaw       = float(yaw)
    ctrl.roll      = float(roll)
    ctrl.jump      = bool(jump >= 0.5)
    ctrl.boost     = bool(boost >= 0.5)
    ctrl.handbrake = bool(handbrake >= 0.5)
    return ctrl

def n_actions():
    return len(_LUT)

# rlbot/actions.py
import os
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState

def load_lookup_table(path: str = "checkpoints/lookup_table.npy"):
    """
    Charge une LUT (N x 8): [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    Si absente, crée une petite LUT par défaut (27 actions utiles).
    """
    if os.path.exists(path):
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] == 8:
            return arr.astype(np.float32)
        print("[actions] lookup_table.npy invalide, fallback 27 actions.")
    # Fallback 27: combinaisons de [-1, 0, 1] sur steer/yaw + throttle on/off + boost on/off + jump on/off
    lut = []
    for throttle in [0.0, 1.0]:
        for steer in [-1.0, 0.0, 1.0]:
            for boost in [0.0, 1.0]:
                for jump in [0.0, 1.0]:
                    # pitch,yaw,roll,handbrake à 0; yaw = steer pour du au sol
                    lut.append([throttle, steer, 0.0, steer, 0.0, jump, boost, 0.0])
    return np.array(lut, dtype=np.float32)

_LUT = load_lookup_table()

def map_action_index(a_idx: int) -> SimpleControllerState:
    a_idx = int(a_idx) % len(_LUT)
    thr, steer, pitch, yaw, roll, jump, boost, handbrake = _LUT[a_idx]
    ctrl = SimpleControllerState()
    ctrl.throttle   = float(thr)
    ctrl.steer      = float(steer)
    ctrl.pitch      = float(pitch)
    ctrl.yaw        = float(yaw)
    ctrl.roll       = float(roll)
    ctrl.jump       = bool(jump >= 0.5)
    ctrl.boost      = bool(boost >= 0.5)
    ctrl.handbrake  = bool(handbrake >= 0.5)
    return ctrl

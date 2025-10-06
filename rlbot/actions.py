import os, numpy as np
from rlbot.agents.base_agent import SimpleControllerState

def _load_lookup_table():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "lookup_table.npy"))
    if os.path.exists(path):
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] == 8:
            print(f"[actions] Loaded LUT: {arr.shape[0]} actions")
            return arr.astype(np.float32)
        print("[actions] lookup_table.npy invalide; fallback.")
    # Fallback 27 actions simples (au sol)
    lut = []
    for throttle in [0.0, 1.0]:
        for steer in [-1.0, 0.0, 1.0]:
            for boost in [0.0, 1.0]:
                for jump in [0.0, 1.0]:
                    lut.append([throttle, steer, 0.0, steer, 0.0, jump, boost, 0.0])
    print("[actions] Fallback LUT: 27 actions")
    return np.array(lut, dtype=np.float32)

_LUT = _load_lookup_table()

def n_actions() -> int:
    return _LUT.shape[0]

def map_action_index(a_idx: int) -> SimpleControllerState:
    a_idx = int(a_idx) % _LUT.shape[0]
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

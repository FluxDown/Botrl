"""
Génère une LookupTable de 90 actions standard pour Rocket League

Basé sur les combinaisons utilisées par RLGym:
- throttle: -1, 0, 1
- steer: -1, -0.5, 0, 0.5, 1
- pitch: -1, 0, 1
- yaw: -1, 0, 1
- roll: 0
- jump: 0, 1
- boost: 0, 1
- handbrake: 0

90 actions = combinaisons les plus utiles
"""

import numpy as np
import os

# Générer les 90 actions standard
# Format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

actions = []

# Base: throttle × steer × (pitch/yaw/jump/boost combos)
for throttle in [-1.0, 0.0, 1.0]:
    for steer in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for pitch in [-1.0, 0.0, 1.0]:
            for boost in [0.0, 1.0]:
                # Ajouter quelques combinaisons clés
                if len(actions) < 90:
                    actions.append([
                        throttle,
                        steer,
                        pitch,
                        steer,  # yaw = steer pour cohérence
                        0.0,     # roll
                        0.0,     # jump
                        boost,
                        0.0      # handbrake
                    ])

# Ajouter des actions de jump si on n'a pas encore 90
while len(actions) < 90:
    actions.append([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # jump

# Prendre exactement 90
actions = actions[:90]

action_space = np.array(actions, dtype=np.float32)

print(f"LookupTable générée:")
print(f"  Shape: {action_space.shape}")
print(f"  Nombre d'actions: {len(action_space)}")
print(f"\nPremières actions:")
for i in range(min(10, len(action_space))):
    thr, steer, pitch, yaw, roll, jump, boost, hb = action_space[i]
    print(f"  {i:2d}: thr={thr:+.1f} steer={steer:+.1f} pitch={pitch:+.1f} boost={boost:.0f}")

# Sauvegarder
output_path = "checkpoints/lookup_table.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, action_space)

print(f"\n✓ Sauvegardé dans: {output_path}")
print(f"  Ce fichier sera utilisé par RLBot pour mapper les actions.")

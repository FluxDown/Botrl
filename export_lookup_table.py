"""
Export LookupTable depuis RLGym vers checkpoints/lookup_table.npy

La LookupTable de RLGym contient 90 actions prédéfinies.
Format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
"""

import numpy as np
import os
from src.utils.worker import create_env
from src.utils.config import load_config

# Créer un env temporaire pour récupérer les actions
config = load_config('config.yaml')
env = create_env(config)

# Récupérer l'action parser
action_parser = env.action_parser

# Parser toutes les 90 actions
actions = []
for i in range(90):
    # Parse l'action i vers les contrôles
    parsed = action_parser.parse_actions(np.array([[i]]), {0: None})
    # parsed est un dict {agent_id: controls}
    controls = parsed[list(parsed.keys())[0]]
    actions.append(controls)

# Convertir en array numpy (90 x 8)
action_space = np.array(actions, dtype=np.float32)

env.close()

print(f"LookupTable shape: {action_space.shape}")
print(f"Nombre d'actions: {len(action_space)}")
print(f"Format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]")
print(f"\nPremières actions:")
for i in range(min(10, len(action_space))):
    print(f"  Action {i}: {action_space[i]}")

# Sauvegarder
output_path = "checkpoints/lookup_table.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, action_space)

print(f"\n✓ Sauvegardé dans: {output_path}")
print(f"  Shape: {action_space.shape}")
print(f"  Dtype: {action_space.dtype}")
print(f"\nCe fichier sera utilisé par RLBot pour mapper les actions du policy.")

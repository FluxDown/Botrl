# Rocket League RL Bot - RLGym 2.0.1

Un bot d'intelligence artificielle complexe pour Rocket League utilisant le reinforcement learning avec RLGym 2.0.1 et PPO.

## Installation

```bash
pip install -r requirements.txt
```

## Structure du projet

```
Botv0.0.31/
├── config.yaml              # Configuration d'entraînement
├── requirements.txt         # Dépendances Python
├── train.py                # Script d'entraînement principal
├── evaluate.py             # Script d'évaluation
├── src/
│   ├── rewards/            # Fonctions de récompense personnalisées
│   ├── obs/                # Observation builders
│   ├── networks/           # Architectures de réseaux neuronaux
│   ├── agents/             # Algorithmes RL (PPO)
│   └── utils/              # Utilitaires
└── checkpoints/            # Modèles sauvegardés
```

## Utilisation

### Entraînement

```bash
python train.py
```

### Évaluation

```bash
python evaluate.py --checkpoint checkpoints/model_best.pth
```

## Fonctionnalités

- **Récompenses complexes**: Goal scoring, aerial play, flip resets, demos, boost management
- **Observations riches**: Positions, vitesses, orientations, boost, état du jeu
- **Architecture PPO optimisée**: Réseau profond avec normalisation
- **Logging avancé**: TensorBoard et Weights & Biases support
- **Checkpointing**: Sauvegarde automatique des meilleurs modèles

## Configuration

Modifiez `config.yaml` pour ajuster les hyperparamètres d'entraînement.

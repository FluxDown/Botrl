# 👁️ Visualisation Live du Bot pendant l'Entraînement

Ce guide explique comment **voir ton bot jouer dans Rocket League** pendant qu'il s'entraîne dans RocketSim.

## 🎯 Principe

- **Process 1** (`train_mp.py`) : Entraînement rapide dans RocketSim (2000-2500 it/s)
  - Sauvegarde `latest_policy.pt` toutes les 10k steps (~4 secondes)

- **Process 2** (RLBot) : Bot qui joue dans Rocket League
  - Recharge automatiquement `latest_policy.pt` toutes les 2 secondes
  - Tu vois ton bot s'améliorer en temps réel !

## 📋 Prérequis

1. **RLBot installé**
   ```bash
   pip install rlbot
   ```

2. **Rocket League** lancé et en mode exhibition/freeplay

3. **Training en cours** (`train_mp.py` doit tourner)

## 🚀 Utilisation

### Étape 1 : Lancer l'entraînement

```bash
python train_mp.py
```

Attends au moins 10k steps pour que `latest_policy.pt` soit créé.

### Étape 2 : Lancer RLBot

**Option A : Via RLBotGUI (recommandé)**
1. Lance RLBotGUI
2. Ajoute le bot : `Add` → Browse vers `rlbot_live_viewer.cfg`
3. Créer un match : ton bot vs psyonix bot
4. Start match

**Option B : Via ligne de commande**
```bash
rlbot gui
```
Puis ajoute le bot manuellement.

### Étape 3 : Observer

- Ouvre Rocket League
- Regarde ton bot jouer
- Toutes les 2 secondes, il recharge les derniers poids
- Tu verras son comportement s'améliorer au fil du temps !

## 🔧 Configuration

### Fréquence de sauvegarde du policy

Dans `train_mp.py` ligne 216 :
```python
if total_steps % 10000 == 0:  # Change 10000 pour ajuster
```

### Fréquence de reload dans le bot

Dans `rlbot_live_viewer.py` ligne 48 :
```python
self.reload_interval = 2.0  # Secondes
```

## ⚠️ Limitations actuelles

**TODO : Implémenter la conversion d'observation**

Le fichier `rlbot_live_viewer.py` contient actuellement un placeholder pour l'observation :

```python
# Ligne 153 - À COMPLÉTER
action_idx = self.policy.act(np.zeros(107))  # Remplacer par vraie obs
```

Il faut convertir `GameTickPacket` (RLBot) → observation RLGym.

Deux approches :

### Option 1 : Conversion manuelle

Extraire depuis `packet`:
- Position/velocity de ta voiture
- Position/velocity de la balle
- Boost amount, has_wheel_contact, etc.
- Organiser dans le même format que `AdvancedObs`

### Option 2 : Utiliser un env RLBot-RLGym bridge

Utiliser un wrapper qui fait la conversion automatiquement.

## 📊 Vérification

Dans la console du bot RLBot, tu devrais voir :
```
[LivePolicy] Reloaded weights @ 50000 steps
[LivePolicy] Reloaded weights @ 60000 steps
...
```

## 🎮 Tips

- **Début d'entraînement** : Le bot sera mauvais (actions random)
- **Après 100k-200k steps** : Tu commenceras à voir des comportements cohérents
- **Après 500k-1M steps** : Touches de balle intentionnelles
- **Après plusieurs millions** : Gameplay avancé

## 🐛 Debug

**Le bot ne bouge pas :**
- Vérifie que `latest_policy.pt` existe dans `checkpoints/`
- Regarde la console RLBot pour les erreurs

**Le bot fait n'importe quoi :**
- Normal en début d'entraînement
- Vérifie que les observations sont correctement normalisées

**Pas de reload :**
- Vérifie que le training tourne et sauvegarde
- Check les timestamps du fichier `latest_policy.pt`

## 🔥 Prochaines étapes

1. Implémenter la conversion d'observation (GameTickPacket → RLGym obs)
2. Tester avec plusieurs bots en match
3. Créer des vidéos timelapses du progrès !

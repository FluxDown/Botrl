# 🚀 QUICKSTART - Lancer l'entraînement MAINTENANT

## ✅ Tout est PRÊT !

Toutes les corrections sont faites. Lance simplement :

```bash
python train_optimized.py
```

---

## 📋 Checklist des corrections appliquées

### ✅ 1. reward_parts propagés
- **Fichier** : `src/utils/env_wrapper.py` (RewardPartsWrapper)
- **Status** : ✅ Corrigé
- **Impact** : TensorBoard affichera maintenant les graphes `reward/goal`, `reward/touch`, etc.

### ✅ 2. Callbacks branchés
- **Fichiers** :
  - `src/utils/custom_callbacks.py` (CustomTBCallback + EvalCallback)
  - `train_optimized.py` (ligne 30-31)
- **Status** : ✅ Branchés dans train_optimized.py
- **Impact** :
  - Logging enrichi (reward breakdown, gradients, success rate)
  - Évaluation auto toutes les 50k steps
  - Sauvegarde best model automatique

### ✅ 3. tick_skip corrigé
- **Fichier** : `config.yaml` (ligne 24)
- **Valeur** : `tick_skip: 8` (était 160 ❌)
- **Status** : ✅ Corrigé
- **Impact** : Signal d'apprentissage 20x meilleur !

### ✅ 4. Throughput PPO aligné
- **Fichier** : `config.yaml` (ligne 6-7)
- **Formule** : `num_envs (8) × n_steps (512) = batch_size (4096)`
- **Status** : ✅ Aligné
- **Impact** : Batches propres, pas de padding, meilleure efficacité GPU

### ✅ 5. CUDA optimisé
- **Fichier** : `train_optimized.py` (ligne 34-49)
- **Optimisations** :
  - ✅ `torch.set_float32_matmul_precision('high')` (TF32 sur Ampere+)
  - ✅ `torch.backends.cudnn.benchmark = True`
  - ✅ `torch.compile(policy)` (PyTorch 2.0+)
- **Status** : ✅ Activé
- **Impact** : -5% à -20% de temps d'entraînement

---

## 🎯 Commandes rapides

### 1. Lancer l'entraînement

```bash
python train_optimized.py
```

### 2. Monitorer en temps réel

**Terminal 2** :
```bash
tensorboard --logdir=./logs
```

Puis ouvre **http://localhost:6006**

### 3. Vérifier CUDA (optionnel)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 📊 Ce que tu vas voir dans TensorBoard

### Graphes disponibles

| Catégorie | Métriques | Description |
|-----------|-----------|-------------|
| **rollout/** | `ep_rew_mean`, `ep_len_mean` | Récompense et longueur moyennes des épisodes |
| **reward/** | `goal_mean`, `touch_mean`, `progress_mean`, `boost_mean`, `demo_mean`, `aerial_mean` | Breakdown des composantes de récompense |
| **reward/** | `goal_hist`, `touch_hist`, etc. | Histogrammes des distributions |
| **train/** | `policy_loss`, `value_loss`, `entropy_loss` | Pertes d'entraînement |
| **train/** | `grad_norm` | Norme des gradients (détecte vanishing/exploding) |
| **metrics/** | `success_rate` | % de buts marqués |
| **eval/** | `mean_reward`, `std_reward` | Performance en évaluation déterministe |

### À surveiller

🟢 **Bon signe** :
- `rollout/ep_rew_mean` augmente
- `reward/goal_mean` > 0 (buts marqués !)
- `metrics/success_rate` > 0.1 (après ~100k steps)
- `train/grad_norm` entre 0.1 et 10

🔴 **Mauvais signe** :
- `train/grad_norm` > 100 → gradients explosent (réduis `learning_rate`)
- `train/grad_norm` < 0.001 → gradients s'annulent (augmente `learning_rate`)
- `rollout/ep_rew_mean` stagne → vérifie les reward weights

---

## ⚙️ Configuration actuelle

```yaml
# Entraînement
total_timesteps: 10M
batch_size: 4096 (8 envs × 512 steps)
learning_rate: 0.0003
n_epochs: 4
seed: 42

# Environnement
num_envs: 8 (auto-ajusté selon CPU)
tick_skip: 8 (15 FPS)
team_size: 1v1

# Réseau
policy_layers: [256, 256, 256]
value_layers: [256, 256, 256]

# Récompenses clés
goal: +100
concede: -100
velocity_ball_to_goal: +2.0
touch: +1.0
```

---

## 🐛 Dépannage rapide

### Erreur : "CUDA out of memory"

```yaml
# config.yaml
training:
  batch_size: 2048  # Au lieu de 4096
  n_steps: 256      # Au lieu de 512

environment:
  num_envs: 4  # Au lieu de 8
```

### Erreur : "rlviser not found"

**C'est normal !** RLViser est optionnel. L'entraînement fonctionne sans.

### Performances lentes (< 1000 steps/sec)

1. Vérifie CUDA : `nvidia-smi`
2. Utilise `train_optimized.py` (pas `train.py`)
3. Réduis `num_envs` si CPU limité

### Reward ne monte pas après 100k steps

1. Vérifie `config.yaml` → `tick_skip: 8` (pas 160 !)
2. Regarde `reward/*/hist` dans TensorBoard
3. Augmente `learning_rate: 0.001` temporairement

---

## 📈 Timeline attendue

| Steps | Comportement attendu |
|-------|---------------------|
| 0-10k | Random, reward ~0 |
| 10k-50k | Commence à toucher la balle |
| 50k-200k | Se dirige vers la balle, premiers buts |
| 200k-500k | Buts réguliers, meilleure défense |
| 500k-1M | Stratégies plus avancées |
| 1M+ | Amélioration continue |

---

## 🎮 Fichiers générés

```
Botrl/
├── checkpoints/
│   ├── best_model.pth       # ⭐ Meilleur modèle (selon éval)
│   ├── checkpoint_100k.pth  # Sauvegardes toutes les 100k
│   ├── checkpoint_200k.pth
│   └── model_final.pth      # Modèle final
│
└── logs/
    └── events.out.tfevents.* # Logs TensorBoard
```

---

## 🚀 GO !

**Tout est prêt. Lance maintenant :**

```bash
python train_optimized.py
```

**Et dans un autre terminal :**

```bash
tensorboard --logdir=./logs
```

**Bon entraînement ! ⚽🤖**

---

## 💡 Tips avancés

### Accélérer encore plus

1. **Augmente num_envs** (si CPU le permet) :
   ```yaml
   environment:
     num_envs: 12
   training:
     n_steps: 341  # Pour garder 12 × 341 ≈ 4096
   ```

2. **Active Weights & Biases** :
   ```yaml
   logging:
     wandb: true
   ```

3. **Teste différents tick_skip** :
   - `tick_skip: 8` → 15 FPS (recommandé)
   - `tick_skip: 12` → 10 FPS (plus rapide à entraîner)
   - `tick_skip: 16` → 7.5 FPS (compromis)

### Tweaker les rewards

Si le bot ne marque pas assez de buts après 500k steps :

```yaml
rewards:
  goal_weight: 200.0  # ⬆️ Double reward
  velocity_ball_to_goal_weight: 5.0  # ⬆️ Encourage plus
```

Si le bot reste collé à la balle sans tirer :

```yaml
rewards:
  velocity_ball_to_goal_weight: 10.0  # ⬆️⬆️ Force à pousser vers le but
  align_ball_goal_weight: 3.0  # ⬆️ Bonus alignement
```

---

**Questions ? Lance et observe les courbes TensorBoard ! 📊**

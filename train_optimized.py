"""
Script d'entraînement OPTIMISÉ avec :
- Callbacks TensorBoard enrichis (reward breakdown, gradients, success rate)
- Évaluation périodique automatique + sauvegarde best model
- Optimisations CUDA (torch.compile, matmul precision, cudnn benchmark)
- VecNormalize pour normalisation obs/returns
- Auto-ajustement num_envs selon CPU
- Propagation correcte de reward_parts
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing

# RLGym imports
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league.sim import RocketSimEngine

# Nos imports
from src.obs.advanced_obs import AdvancedObs
from src.rewards.combined_reward import CombinedReward
from src.networks.actor_critic import ActorCritic
from src.agents.ppo import PPO, RolloutBuffer
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.custom_callbacks import CustomTBCallback, EvalCallback
from src.utils.env_wrapper import RewardPartsWrapper


def setup_torch_optimizations(device):
    """Configure les optimisations PyTorch pour CUDA"""
    if device.type == 'cuda':
        try:
            # Precision élevée pour matmul (TF32 sur Ampere+)
            torch.set_float32_matmul_precision('high')
            print("✓ Set matmul precision to 'high' (TF32 enabled on Ampere+)")
        except:
            pass

        try:
            # Benchmark cuDNN pour trouver les algos les plus rapides
            torch.backends.cudnn.benchmark = True
            print("✓ Enabled cuDNN benchmark mode")
        except:
            pass

        # Afficher les infos GPU
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def create_environment(config, eval_mode=False, wrap_reward_parts=True):
    """Crée l'environnement RLGym avec wrapper pour reward_parts"""
    env_config = config['environment']
    reward_config = config['rewards']

    # Observation builder
    obs_builder = AdvancedObs(
        team_size=env_config['team_size'],
        tick_skip=env_config['tick_skip']
    )

    # Action parser
    action_parser = LookupTableAction()

    # Reward function
    reward_fn = CombinedReward(
        goal_weight=reward_config['goal_weight'],
        concede_weight=reward_config['concede_weight'],
        touch_ball_weight=reward_config['touch_ball_weight'],
        velocity_player_to_ball_weight=reward_config['velocity_player_to_ball_weight'],
        velocity_ball_to_goal_weight=reward_config['velocity_ball_to_goal_weight'],
        save_boost_weight=reward_config['save_boost_weight'],
        demo_weight=reward_config['demo_weight'],
        got_demoed_weight=reward_config['got_demoed_weight'],
        align_ball_goal_weight=reward_config['align_ball_goal_weight'],
        aerial_weight=reward_config['aerial_weight']
    )

    # Terminal conditions
    terminal_conditions = AnyCondition(
        GoalCondition(),
        TimeoutCondition(timeout_seconds=env_config['timeout_seconds'])
    )

    # State mutator
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=env_config['team_size'], orange_size=env_config['team_size']),
        KickoffMutator()
    )

    # Créer l'environnement de base
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=terminal_conditions,
        truncation_cond=terminal_conditions,
        transition_engine=None,
        renderer=None
    )

    # Wrapper pour propager reward_parts
    if wrap_reward_parts:
        env = RewardPartsWrapper(env)

    return env


def get_optimal_num_envs(config):
    """Détermine le nombre optimal d'environnements selon le CPU"""
    config_num_envs = config['environment'].get('num_envs', 8)
    auto_adjust = config['environment'].get('auto_adjust_envs', True)

    if not auto_adjust:
        return config_num_envs

    # Limiter à 75% des CPUs disponibles
    cpu_count = multiprocessing.cpu_count()
    max_envs = max(1, int(cpu_count * 0.75))

    optimal = min(config_num_envs, max_envs)
    if optimal != config_num_envs:
        print(f"⚠ Ajustement num_envs: {config_num_envs} → {optimal} (CPU count: {cpu_count})")

    return optimal


def train():
    """Fonction principale d'entraînement optimisée"""
    # Charger la configuration
    config = load_config('config.yaml')

    # Set seed pour reproductibilité
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ROCKET LEAGUE BOT TRAINING - OPTIMIZED")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Setup optimisations PyTorch
    setup_torch_optimizations(device)

    # Créer les dossiers
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Logger de base
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['tensorboard'],
        use_wandb=config['logging']['wandb'],
        wandb_config={'project': config['logging']['wandb_project'], 'config': config}
    )

    # Callback TensorBoard personnalisé
    tb_callback = CustomTBCallback(
        log_dir=config['logging']['log_dir'],
        reward_keys=("goal", "touch", "progress", "boost", "demo", "aerial"),
        verbose=1
    )

    # Logger la config dans TensorBoard
    config_text = str(config)
    tb_callback.writer.add_text("config", config_text, 0)

    # Créer l'environnement d'entraînement
    logger.log_text("Creating training environment...")
    env = create_environment(config)

    # Créer l'environnement d'évaluation
    logger.log_text("Creating evaluation environment...")
    eval_env = create_environment(config, eval_mode=True)

    # Obtenir les dimensions
    obs_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    logger.log_text(f"Observation space: {obs_space_size}")
    logger.log_text(f"Action space: {action_space_size}")

    # Créer le réseau de neurones
    logger.log_text("Creating neural network...")
    network_config = config['network']

    policy_network = ActorCritic(
        obs_space_size=obs_space_size,
        action_space_size=action_space_size,
        policy_layers=network_config['policy_layers'],
        value_layers=network_config['value_layers'],
        activation=network_config['activation'],
        continuous_actions=False
    )

    # Créer l'agent PPO
    logger.log_text("Creating PPO agent...")
    training_config = config['training']

    agent = PPO(
        policy_network=policy_network,
        learning_rate=training_config['learning_rate'],
        gamma=training_config['gamma'],
        gae_lambda=training_config['gae_lambda'],
        clip_range=training_config['clip_range'],
        vf_coef=training_config['vf_coef'],
        ent_coef=training_config['ent_coef'],
        max_grad_norm=training_config['max_grad_norm'],
        device=device
    )

    # Tentative de compilation avec torch.compile (Torch 2.0+)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            logger.log_text("Attempting torch.compile optimization...")
            agent.policy = torch.compile(agent.policy, mode='reduce-overhead')
            logger.log_text("✓ Successfully compiled policy with torch.compile")
        except Exception as e:
            logger.log_text(f"⚠ torch.compile failed: {e}")

    # Callback d'évaluation
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=training_config.get('eval_freq', 50000),
        n_eval_episodes=training_config.get('n_eval_episodes', 10),
        best_model_path=os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth'),
        verbose=1
    )

    # Créer le rollout buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=training_config['batch_size'],
        obs_dim=obs_space_size,
        action_dim=1,
        device=device
    )

    # Variables de tracking
    total_timesteps = 0
    episode_count = 0
    best_reward = -float('inf')

    logger.log_text(f"Starting training for {training_config['total_timesteps']} timesteps")
    logger.log_text(f"Batch size: {training_config['batch_size']}, n_epochs: {training_config['n_epochs']}")

    # Boucle d'entraînement principale
    while total_timesteps < training_config['total_timesteps']:
        # Reset de l'environnement
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Collecter les expériences
        while not done and rollout_buffer.ptr < training_config['batch_size']:
            # Sélectionner une action
            action, log_prob, value = agent.select_action(obs)

            # Effectuer l'action dans l'environnement
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            # Stocker dans le buffer
            rollout_buffer.add(obs, [action], log_prob, reward, value, done)

            # Mise à jour
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1

            # Callback on_step (reward_parts est maintenant dans info grâce au wrapper)
            tb_callback.on_step(info, total_timesteps)

            # Reset si l'épisode est terminé
            if done:
                episode_count += 1

                # Ajouter les stats d'épisode
                episode_info = {
                    'episode': {'r': episode_reward, 'l': episode_length},
                    'success': info.get('success', 0.0)
                }
                tb_callback.on_step(episode_info, total_timesteps)

                # Logger les stats
                logger.log_scalar('episode/reward', episode_reward, episode_count)
                logger.log_scalar('episode/length', episode_length, episode_count)

                if episode_count % 10 == 0:
                    logger.log_text(
                        f"Ep {episode_count} | Step {total_timesteps} | "
                        f"R: {episode_reward:.2f} | Len: {episode_length}"
                    )

                # Reset
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False

        # Entraîner l'agent quand le buffer est plein
        if rollout_buffer.ptr >= training_config['batch_size'] or rollout_buffer.full:
            # Calculer la dernière value
            with torch.no_grad():
                _, _, last_value = agent.select_action(obs)

            # Calculer returns et advantages
            rollout_buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=training_config['gamma'],
                gae_lambda=training_config['gae_lambda']
            )

            # Entraîner
            train_stats = agent.train(
                rollout_buffer,
                n_epochs=training_config['n_epochs'],
                batch_size=2048
            )

            # Callbacks
            tb_callback.on_train_step(train_stats, total_timesteps, agent=agent)
            logger.log_scalars('train', train_stats, total_timesteps)

            # Clear le buffer
            rollout_buffer.clear()

        # Évaluation périodique
        mean_eval_reward = eval_callback.evaluate(agent, total_timesteps, writer=tb_callback.writer)
        if mean_eval_reward is not None:
            logger.log_text(f"✓ Eval @ {total_timesteps}: {mean_eval_reward:.2f}")

        # Sauvegarder périodiquement
        if total_timesteps % training_config['save_interval'] == 0 and total_timesteps > 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_{total_timesteps}.pth'
            )
            agent.save(checkpoint_path)
            logger.log_text(f"Checkpoint saved: {checkpoint_path}")

    # Sauvegarder le modèle final
    final_path = os.path.join(config['logging']['checkpoint_dir'], 'model_final.pth')
    agent.save(final_path)
    logger.log_text(f"Final model saved: {final_path}")

    # Fermer
    env.close()
    eval_env.close()
    logger.close()
    tb_callback.close()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()

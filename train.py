"""
Script d'entraînement principal pour le bot Rocket League avec RLGym 2.0.1
"""

import os
import torch
import numpy as np
from tqdm import tqdm

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


def create_environment(config):
    """Crée l'environnement RLGym"""
    env_config = config['environment']
    reward_config = config['rewards']

    # Observation builder
    obs_builder = AdvancedObs(
        team_size=env_config['team_size'],
        tick_skip=env_config['tick_skip']
    )

    # Action parser - Utilise une lookup table discrète
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

    # State mutator (pour varier les conditions de départ)
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=env_config['team_size'], orange_size=env_config['team_size']),
        KickoffMutator()
    )

    # Créer l'environnement
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=terminal_conditions,
        truncation_cond=terminal_conditions,
        transition_engine=None,  # Utilise le moteur par défaut
        renderer=None  # Pas de rendu pendant l'entraînement
    )

    return env


def train():
    """Fonction principale d'entraînement"""
    # Charger la configuration
    config = load_config('config.yaml')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Créer les dossiers
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['tensorboard'],
        use_wandb=config['logging']['wandb'],
        wandb_config={'project': config['logging']['wandb_project'], 'config': config}
    )

    logger.log_text("=" * 50)
    logger.log_text("Starting Rocket League Bot Training")
    logger.log_text("=" * 50)

    # Créer l'environnement
    logger.log_text("Creating environment...")
    env = create_environment(config)

    # Obtenir les dimensions
    obs_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n  # Nombre d'actions discrètes

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
        continuous_actions=False  # Actions discrètes avec LookupTable
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

    # Créer le rollout buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=training_config['batch_size'],
        obs_dim=obs_space_size,
        action_dim=1,  # Actions discrètes = 1 dimension
        device=device
    )

    # Variables de tracking
    total_timesteps = 0
    episode_count = 0
    best_reward = -float('inf')

    logger.log_text(f"Starting training for {training_config['total_timesteps']} timesteps")

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

            # Reset si l'épisode est terminé
            if done:
                episode_count += 1

                # Logger les stats de l'épisode
                logger.log_scalar('episode/reward', episode_reward, episode_count)
                logger.log_scalar('episode/length', episode_length, episode_count)
                logger.log_scalar('training/total_timesteps', total_timesteps, episode_count)

                if episode_count % 10 == 0:
                    logger.log_text(
                        f"Episode {episode_count} | Timesteps: {total_timesteps} | "
                        f"Reward: {episode_reward:.2f} | Length: {episode_length}"
                    )

                # Reset pour le prochain épisode
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
                done = False

        # Entraîner l'agent quand le buffer est plein
        if rollout_buffer.ptr >= training_config['batch_size'] or rollout_buffer.full:
            # Calculer la dernière value pour GAE
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

            # Logger les stats d'entraînement
            logger.log_scalars('train', train_stats, total_timesteps)

            # Clear le buffer
            rollout_buffer.clear()

        # Sauvegarder périodiquement
        if total_timesteps % training_config['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_{total_timesteps}.pth'
            )
            agent.save(checkpoint_path)
            logger.log_text(f"Checkpoint saved: {checkpoint_path}")

            # Sauvegarder le meilleur modèle
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = os.path.join(config['logging']['checkpoint_dir'], 'model_best.pth')
                agent.save(best_path)
                logger.log_text(f"New best model saved with reward: {best_reward:.2f}")

    # Sauvegarder le modèle final
    final_path = os.path.join(config['logging']['checkpoint_dir'], 'model_final.pth')
    agent.save(final_path)
    logger.log_text(f"Final model saved: {final_path}")

    # Fermer
    env.close()
    logger.close()

    logger.log_text("=" * 50)
    logger.log_text("Training completed!")
    logger.log_text("=" * 50)


if __name__ == "__main__":
    train()

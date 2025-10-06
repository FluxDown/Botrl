"""
Script d'évaluation pour tester le bot entraîné
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

# RLGym imports
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator

# Nos imports
from src.obs.advanced_obs import AdvancedObs
from src.rewards.combined_reward import CombinedReward
from src.networks.actor_critic import ActorCritic
from src.agents.ppo import PPO
from src.utils.config import load_config


def create_environment(config, render=True):
    """Crée l'environnement RLGym pour l'évaluation"""
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

    # Créer l'environnement
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=terminal_conditions,
        truncation_cond=terminal_conditions,
        transition_engine=None,
        renderer='rlviser' if render else None  # Rendu visuel si demandé
    )

    return env


def evaluate(checkpoint_path, n_episodes=10, render=True):
    """
    Évalue le bot sur plusieurs épisodes

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        n_episodes: Nombre d'épisodes d'évaluation
        render: Si True, affiche le rendu visuel
    """
    # Charger la configuration
    config = load_config('config.yaml')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Créer l'environnement
    print("Creating environment...")
    env = create_environment(config, render=render)

    # Obtenir les dimensions
    obs_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    # Créer le réseau de neurones
    print("Creating neural network...")
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
    agent = PPO(
        policy_network=policy_network,
        device=device
    )

    # Charger le checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    # Mettre le réseau en mode évaluation
    agent.policy.eval()

    # Statistiques
    episode_rewards = []
    episode_lengths = []
    goals_scored = []
    goals_conceded = []

    print(f"\nEvaluating for {n_episodes} episodes...")

    # Évaluation
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        initial_blue_score = 0
        initial_orange_score = 0

        while not done:
            # Sélectionner une action (déterministe pour l'évaluation)
            action, _, _ = agent.select_action(obs, deterministic=True)

            # Effectuer l'action
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            # Mise à jour
            obs = next_obs
            episode_reward += reward
            episode_length += 1

        # Enregistrer les stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Afficher les stats de l'épisode
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")

    # Statistiques finales
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"Number of episodes: {n_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 50)

    # Fermer l'environnement
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Rocket League RL Bot')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/model_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering'
    )

    args = parser.parse_args()

    # Vérifier que le checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Évaluer
    evaluate(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()

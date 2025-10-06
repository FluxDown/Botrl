"""
Script d'entraînement PARALLÈLE OPTIMISÉ avec :
- 8-12 environnements simultanés (selon CPU)
- Callbacks TensorBoard enrichis (reward breakdown, gradients, success rate)
- Évaluation périodique automatique + sauvegarde best model
- Optimisations CUDA (torch.compile, matmul precision, cudnn benchmark)
- Auto-ajustement num_envs selon CPU
- Propagation correcte de reward_parts
- Throughput aligné (num_envs × n_steps = batch_size)
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
            torch.set_float32_matmul_precision('high')
            print("✓ TF32 matmul precision enabled")
        except:
            pass

        try:
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN benchmark enabled")
        except:
            pass

        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA: {torch.version.cuda}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU Memory: {mem_gb:.2f} GB")


def create_single_env(config, use_renderer=False):
    """Crée un seul environnement RLGym avec wrapper"""
    env_config = config['environment']
    reward_config = config['rewards']

    obs_builder = AdvancedObs(
        team_size=env_config['team_size'],
        tick_skip=env_config['tick_skip']
    )

    action_parser = LookupTableAction()

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

    terminal_conditions = AnyCondition(
        GoalCondition(),
        TimeoutCondition(timeout_seconds=env_config['timeout_seconds'])
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=env_config['team_size'], orange_size=env_config['team_size']),
        KickoffMutator()
    )

    transition_engine = RocketSimEngine()

    renderer = None
    if use_renderer:
        try:
            from src.utils.rlviser_renderer import RLViserRenderer
            renderer = RLViserRenderer()
            print("✓ RLViser enabled at http://localhost:8080")
        except:
            pass

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=terminal_conditions,
        truncation_cond=terminal_conditions,
        transition_engine=transition_engine,
        renderer=renderer
    )

    # Wrapper pour propager reward_parts
    env = RewardPartsWrapper(env)
    return env


class ParallelEnvironments:
    """Gestionnaire d'environnements parallèles optimisé"""

    def __init__(self, config, num_envs=8, render_first=False):
        self.config = config
        self.num_envs = num_envs

        print(f"Creating {num_envs} parallel environments...")
        if render_first:
            print("Environment #0 will have RLViser visualization")

        self.envs = []
        for i in range(num_envs):
            use_render = (i == 0 and render_first)
            env = create_single_env(config, use_renderer=use_render)
            self.envs.append(env)

        self.current_obs = [None] * num_envs
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
        self.agent_ids = None
        self.has_renderer = render_first

    def render_first_env(self):
        """Rend le premier environnement si RLViser est activé"""
        if self.has_renderer and len(self.envs) > 0:
            try:
                self.envs[0].render()
            except:
                pass

    def reset(self, env_id=None):
        """Reset un ou tous les environnements"""
        if env_id is not None:
            result = self.envs[env_id].reset()
            if isinstance(result, tuple):
                obs_dict = result[0]
            else:
                obs_dict = result

            if self.agent_ids is None and isinstance(obs_dict, dict):
                self.agent_ids = list(obs_dict.keys())

            if isinstance(obs_dict, dict):
                obs_list = list(obs_dict.values())
            else:
                obs_list = [obs_dict]

            self.current_obs[env_id] = obs_list
            self.episode_rewards[env_id] = 0.0
            self.episode_lengths[env_id] = 0
            return obs_list
        else:
            for i in range(self.num_envs):
                result = self.envs[i].reset()
                if isinstance(result, tuple):
                    obs_dict = result[0]
                else:
                    obs_dict = result

                if self.agent_ids is None and isinstance(obs_dict, dict):
                    self.agent_ids = list(obs_dict.keys())

                if isinstance(obs_dict, dict):
                    obs_list = list(obs_dict.values())
                else:
                    obs_list = [obs_dict]

                self.current_obs[i] = obs_list
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
            return self.current_obs

    def step(self, actions):
        """Exécute une action dans chaque environnement"""
        observations = []
        rewards = []
        dones = []
        infos = []

        for i in range(self.num_envs):
            action_dict = {}
            for agent_idx, agent_id in enumerate(self.agent_ids):
                action_dict[agent_id] = np.array([int(actions[i][agent_idx])])

            step_result = self.envs[i].step(action_dict)

            if len(step_result) == 5:
                obs_dict, reward_dict, terminated_dict, truncated_dict, info = step_result
            else:
                obs_dict, reward_dict, terminated_dict, truncated_dict = step_result
                info = {}

            obs_list = list(obs_dict.values())
            reward_list = list(reward_dict.values())
            terminated_list = list(terminated_dict.values())
            truncated_list = list(truncated_dict.values())

            done = any(terminated_list) or any(truncated_list)
            mean_reward = np.mean(reward_list)

            observations.append(obs_list)
            rewards.append(mean_reward)
            dones.append(done)
            infos.append(info)

            self.episode_rewards[i] += mean_reward
            self.episode_lengths[i] += 1

            # Auto-reset si terminé
            if done:
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_lengths[i]
                }
                # Marquer succès (reward_parts contient goal info via wrapper)
                info['success'] = info.get('success', 0.0)

                result = self.envs[i].reset()
                if isinstance(result, tuple):
                    obs_dict = result[0]
                else:
                    obs_dict = result

                if isinstance(obs_dict, dict):
                    obs_list = list(obs_dict.values())
                else:
                    obs_list = [obs_dict]

                observations[i] = obs_list
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

            self.current_obs[i] = observations[i]

        return observations, rewards, dones, infos

    def close(self):
        """Ferme tous les environnements"""
        for env in self.envs:
            env.close()

    def get_obs_space(self):
        """Retourne la dimension de l'espace d'observation"""
        obs_space = self.envs[0].observation_space
        if callable(obs_space):
            obs_dict, _ = self.envs[0].reset()
            if isinstance(obs_dict, dict):
                test_obs = list(obs_dict.values())[0]
            else:
                test_obs = obs_dict
            return len(test_obs)
        else:
            return obs_space.shape[0]

    def get_action_space(self):
        """Retourne le nombre d'actions discrètes"""
        action_space = self.envs[0].action_space
        if hasattr(action_space, 'n'):
            return action_space.n
        else:
            return 90


def get_optimal_num_envs(config):
    """Détermine le nombre optimal d'environnements selon le CPU"""
    config_num_envs = config['environment'].get('num_envs', 8)
    auto_adjust = config['environment'].get('auto_adjust_envs', True)

    if not auto_adjust:
        return config_num_envs

    cpu_count = multiprocessing.cpu_count()
    max_envs = max(1, int(cpu_count * 0.75))

    optimal = min(config_num_envs, max_envs)
    if optimal != config_num_envs:
        print(f"⚠ Auto-adjust num_envs: {config_num_envs} → {optimal} (CPU: {cpu_count})")

    return optimal


def train():
    """Fonction principale d'entraînement parallèle optimisé"""
    config = load_config('config.yaml')

    # Seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ROCKET LEAGUE BOT - PARALLEL TRAINING (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Setup optimisations PyTorch
    setup_torch_optimizations(device)

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

    # Callback TensorBoard
    tb_callback = CustomTBCallback(
        log_dir=config['logging']['log_dir'],
        reward_keys=("goal", "touch", "progress", "boost", "demo", "aerial"),
        verbose=1
    )

    # Logger la config
    config_text = str(config)
    tb_callback.writer.add_text("config", config_text, 0)

    # Nombre d'environnements optimal
    num_envs = get_optimal_num_envs(config)
    logger.log_text(f"Using {num_envs} parallel environments")

    # Créer les environnements parallèles
    logger.log_text("Creating parallel environments...")
    envs = ParallelEnvironments(config, num_envs=num_envs, render_first=False)

    # Créer environnement d'évaluation
    logger.log_text("Creating evaluation environment...")
    eval_env = create_single_env(config, use_renderer=False)

    # Obtenir les dimensions
    test_obs_list = envs.reset()
    obs_space_size = len(test_obs_list[0][0])
    action_space_size = envs.get_action_space()

    logger.log_text(f"Observation space: {obs_space_size}")
    logger.log_text(f"Action space: {action_space_size}")
    logger.log_text(f"Agents per env: {len(test_obs_list[0])}")

    # Créer le réseau
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

    # torch.compile si CUDA
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            agent.policy = torch.compile(agent.policy, mode='reduce-overhead')
            logger.log_text("✓ torch.compile enabled")
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

    # Rollout buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=training_config['batch_size'],
        obs_dim=obs_space_size,
        action_dim=1,
        device=device
    )

    # Variables de tracking
    total_timesteps = 0
    episode_count = 0
    episode_rewards_buffer = []

    logger.log_text(f"Starting training for {training_config['total_timesteps']} timesteps")
    logger.log_text(f"Batch size: {training_config['batch_size']} ({num_envs} envs × {training_config.get('n_steps', 512)} steps)")

    # Reset initial
    current_obs = envs.reset()

    # Boucle principale
    pbar = tqdm(total=training_config['total_timesteps'], desc="Training")

    while total_timesteps < training_config['total_timesteps']:
        # Collecter les expériences
        while rollout_buffer.ptr < training_config['batch_size']:
            actions = []
            log_probs = []
            values = []

            for env_obs_list in current_obs:
                env_actions = []
                env_log_probs = []
                env_values = []

                for agent_obs in env_obs_list:
                    action, log_prob, value = agent.select_action(agent_obs)
                    env_actions.append(action)
                    env_log_probs.append(log_prob)
                    env_values.append(value)

                actions.append(env_actions)
                log_probs.append(env_log_probs)
                values.append(env_values)

            # Exécuter les actions
            next_obs, rewards, dones, infos = envs.step(actions)
            envs.render_first_env()

            # Stocker et logger
            for i in range(num_envs):
                for agent_idx in range(len(current_obs[i])):
                    rollout_buffer.add(
                        current_obs[i][agent_idx],
                        [actions[i][agent_idx]],
                        log_probs[i][agent_idx],
                        rewards[i],
                        values[i][agent_idx],
                        dones[i]
                    )

                total_timesteps += 1
                pbar.update(1)

                # Callback (reward_parts maintenant dans infos via wrapper)
                tb_callback.on_step(infos[i], total_timesteps)

                # Logger épisodes terminés
                if dones[i] and 'episode' in infos[i]:
                    episode_count += 1
                    ep_reward = infos[i]['episode']['r']
                    ep_length = infos[i]['episode']['l']

                    episode_rewards_buffer.append(ep_reward)

                    logger.log_scalar('episode/reward', ep_reward, episode_count)
                    logger.log_scalar('episode/length', ep_length, episode_count)

                    if episode_count % 10 == 0:
                        avg_reward = np.mean(episode_rewards_buffer[-100:]) if episode_rewards_buffer else 0
                        logger.log_text(
                            f"Ep {episode_count} | Step {total_timesteps} | "
                            f"R: {ep_reward:.2f} | Avg100: {avg_reward:.2f}"
                        )

            current_obs = next_obs

            if rollout_buffer.ptr >= training_config['batch_size']:
                break

        # Entraîner
        last_values = []
        for env_obs_list in current_obs:
            for agent_obs in env_obs_list:
                with torch.no_grad():
                    _, _, last_value = agent.select_action(agent_obs)
                    last_values.append(last_value)
        avg_last_value = np.mean(last_values)

        rollout_buffer.compute_returns_and_advantages(
            last_value=avg_last_value,
            gamma=training_config['gamma'],
            gae_lambda=training_config['gae_lambda']
        )

        train_stats = agent.train(
            rollout_buffer,
            n_epochs=training_config['n_epochs'],
            batch_size=2048
        )

        # Callbacks
        tb_callback.on_train_step(train_stats, total_timesteps, agent=agent)
        logger.log_scalars('train', train_stats, total_timesteps)

        rollout_buffer.clear()

        # Évaluation
        mean_eval_reward = eval_callback.evaluate(agent, total_timesteps, writer=tb_callback.writer)
        if mean_eval_reward is not None:
            logger.log_text(f"✓ Eval @ {total_timesteps}: {mean_eval_reward:.2f}")

        # Sauvegarder
        if total_timesteps % training_config['save_interval'] == 0 and total_timesteps > 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_{total_timesteps}.pth'
            )
            agent.save(checkpoint_path)
            logger.log_text(f"Checkpoint: {checkpoint_path}")

    pbar.close()

    # Sauvegarder final
    final_path = os.path.join(config['logging']['checkpoint_dir'], 'model_final.pth')
    agent.save(final_path)
    logger.log_text(f"Final model: {final_path}")

    # Fermer
    envs.close()
    eval_env.close()
    logger.close()
    tb_callback.close()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()

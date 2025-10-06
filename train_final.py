"""
🚀 SCRIPT D'ENTRAÎNEMENT FINAL - TOUTES OPTIMISATIONS

Inclut :
✅ Multi-environnements parallèles (8-12 selon CPU)
✅ RewardPartsToInfo wrapper (10 lignes, propage reward_parts)
✅ VecNormalize (obs + returns, sauvegarde .pkl)
✅ LR Schedule (linéaire 3e-4 → 5e-5)
✅ Entropy schedule (0.01 → 0.001)
✅ CUDA optimisé (torch.compile, TF32, cuDNN)
✅ EvalCallback + best model auto
✅ TensorBoard enrichi (reward breakdown, gradients, success)
✅ Throughput aligné (num_envs × n_steps = batch_size)
✅ Logs LR/entropy dans TensorBoard

Config recommandée :
- num_envs: 8-12
- n_steps: 512 (8×512=4096)
- batch_size: 4096
- n_epochs: 4
- gamma: 0.995
- initial_lr: 3e-4
- final_lr: 5e-5
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing

# RLGym
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
from src.utils.vec_wrapper import RewardPartsToInfo, SimpleVecNormalize, LRScheduler


def setup_cuda(device):
    """Setup CUDA optimisations"""
    if device.type != 'cuda':
        return

    try:
        torch.set_float32_matmul_precision('high')
        print("✓ TF32 enabled")
    except:
        pass

    try:
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark")
    except:
        pass

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


def create_env(config):
    """Crée 1 env RLGym avec RewardPartsToInfo wrapper"""
    env_config = config['environment']
    reward_config = config['rewards']

    obs_builder = AdvancedObs(
        team_size=env_config['team_size'],
        tick_skip=env_config['tick_skip']
    )

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

    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=env_config['team_size'], orange_size=env_config['team_size']),
            KickoffMutator()
        ),
        obs_builder=obs_builder,
        action_parser=LookupTableAction(),
        reward_fn=reward_fn,
        termination_cond=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=env_config['timeout_seconds'])
        ),
        truncation_cond=AnyCondition(
            GoalCondition(),
            TimeoutCondition(timeout_seconds=env_config['timeout_seconds'])
        ),
        transition_engine=RocketSimEngine(),
        renderer=None
    )

    # Wrapper reward_parts (10 lignes)
    env = RewardPartsToInfo(env)
    return env


class ParallelEnvs:
    """Multi-environnements avec VecNormalize"""

    def __init__(self, config, num_envs, use_norm=True):
        self.num_envs = num_envs
        self.envs = [create_env(config) for _ in range(num_envs)]
        self.normalizer = SimpleVecNormalize(gamma=config['training']['gamma']) if use_norm else None

        # States
        self.current_obs = [None] * num_envs
        self.ep_rewards = [0.0] * num_envs
        self.ep_lengths = [0] * num_envs
        self.agent_ids = None

    def reset(self):
        obs_list = []
        for i, env in enumerate(self.envs):
            result = env.reset()
            obs_dict = result[0] if isinstance(result, tuple) else result

            if self.agent_ids is None and isinstance(obs_dict, dict):
                self.agent_ids = list(obs_dict.keys())

            obs = list(obs_dict.values()) if isinstance(obs_dict, dict) else [obs_dict]
            self.current_obs[i] = obs
            self.ep_rewards[i] = 0.0
            self.ep_lengths[i] = 0
            obs_list.append(obs[0])  # Premier agent

        obs_array = np.array(obs_list)
        if self.normalizer:
            obs_array = self.normalizer.normalize_obs_array(obs_array)

        return [obs_array[i] for i in range(self.num_envs)]

    def step(self, actions):
        observations = []
        rewards = []
        dones_array = []
        infos = []

        for i in range(self.num_envs):
            action_dict = {self.agent_ids[0]: np.array([int(actions[i])])}
            result = self.envs[i].step(action_dict)

            if len(result) == 5:
                obs_dict, rew_dict, term_dict, trunc_dict, info = result
            else:
                obs_dict, rew_dict, term_dict, trunc_dict = result
                info = {}

            obs = list(obs_dict.values())[0]
            rew = list(rew_dict.values())[0]
            done = list(term_dict.values())[0] or list(trunc_dict.values())[0]

            self.ep_rewards[i] += rew
            self.ep_lengths[i] += 1

            if done:
                info['episode'] = {'r': self.ep_rewards[i], 'l': self.ep_lengths[i]}
                info['success'] = 1.0 if self.ep_rewards[i] > 50 else 0.0

                result_reset = self.envs[i].reset()
                obs = list(result_reset[0].values())[0] if isinstance(result_reset, tuple) else list(result_reset.values())[0]

                self.ep_rewards[i] = 0.0
                self.ep_lengths[i] = 0

            observations.append(obs)
            rewards.append(rew)
            dones_array.append(done)
            infos.append(info)

        obs_array = np.array(observations)
        rew_array = np.array(rewards)
        done_array = np.array(dones_array)

        if self.normalizer:
            obs_array = self.normalizer.normalize_obs_array(obs_array)
            rew_array = self.normalizer.normalize_reward(rew_array, done_array)

        return [obs_array[i] for i in range(self.num_envs)], rew_array.tolist(), done_array.tolist(), infos

    def close(self):
        for env in self.envs:
            env.close()


def get_num_envs(config):
    """Auto-ajuste num_envs"""
    requested = config['environment'].get('num_envs', 8)
    auto = config['environment'].get('auto_adjust_envs', True)

    if not auto:
        return requested

    cpu_count = multiprocessing.cpu_count()
    optimal = min(requested, max(1, int(cpu_count * 0.75)))

    if optimal != requested:
        print(f"⚠ num_envs: {requested} → {optimal} (CPU: {cpu_count})")

    return optimal


def train():
    """TRAINING FINAL avec toutes optimisations"""
    config = load_config('config.yaml')

    # Seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("🚀 ROCKET LEAGUE BOT - FINAL TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")

    setup_cuda(device)

    # Dirs
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['tensorboard'],
        use_wandb=config['logging']['wandb'],
        wandb_config={'project': config['logging']['wandb_project'], 'config': config}
    )

    # TensorBoard callback
    tb_cb = CustomTBCallback(
        log_dir=config['logging']['log_dir'],
        reward_keys=("goal", "touch", "progress", "boost", "demo", "aerial"),
        verbose=1
    )
    tb_cb.writer.add_text("config", str(config), 0)

    # Envs
    num_envs = get_num_envs(config)
    print(f"✓ {num_envs} parallel environments")

    envs = ParallelEnvs(config, num_envs, use_norm=True)
    eval_env = create_env(config)

    # Get dims
    test_obs = envs.reset()
    obs_size = len(test_obs[0])
    act_size = 90  # LookupTable default

    print(f"✓ Obs: {obs_size}, Actions: {act_size}")

    # Network
    net_cfg = config['network']
    policy = ActorCritic(
        obs_space_size=obs_size,
        action_space_size=act_size,
        policy_layers=net_cfg['policy_layers'],
        value_layers=net_cfg['value_layers'],
        activation=net_cfg['activation'],
        continuous_actions=False
    )

    # PPO
    train_cfg = config['training']
    agent = PPO(
        policy_network=policy,
        learning_rate=train_cfg['learning_rate'],
        gamma=train_cfg['gamma'],
        gae_lambda=train_cfg['gae_lambda'],
        clip_range=train_cfg['clip_range'],
        vf_coef=train_cfg['vf_coef'],
        ent_coef=train_cfg['ent_coef'],
        max_grad_norm=train_cfg['max_grad_norm'],
        device=device
    )

    # torch.compile
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            agent.policy = torch.compile(agent.policy, mode='reduce-overhead')
            print("✓ torch.compile enabled")
        except Exception as e:
            print(f"⚠ torch.compile failed: {e}")

    # LR Scheduler
    total_updates = train_cfg['total_timesteps'] // train_cfg['batch_size']
    lr_scheduler = LRScheduler(
        agent.optimizer,
        initial_lr=train_cfg['learning_rate'],
        final_lr=train_cfg.get('final_lr', 5e-5),
        total_steps=total_updates,
        schedule_type='linear'
    )

    # Eval callback
    eval_cb = EvalCallback(
        eval_env=eval_env,
        eval_freq=train_cfg.get('eval_freq', 50000),
        n_eval_episodes=train_cfg.get('n_eval_episodes', 10),
        best_model_path=os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth'),
        verbose=1
    )

    # Buffer
    buffer = RolloutBuffer(
        buffer_size=train_cfg['batch_size'],
        obs_dim=obs_size,
        action_dim=1,
        device=device
    )

    # Vars
    total_steps = 0
    ep_count = 0
    ep_buffer = []

    n_steps = train_cfg.get('n_steps', train_cfg['batch_size'] // num_envs)
    print(f"✓ Batch: {train_cfg['batch_size']} ({num_envs} envs × {n_steps} steps)")
    print(f"✓ LR: {train_cfg['learning_rate']:.0e} → {train_cfg.get('final_lr', 5e-5):.0e}")

    # Reset
    obs_list = envs.reset()

    pbar = tqdm(total=train_cfg['total_timesteps'], desc="Training")

    while total_steps < train_cfg['total_timesteps']:
        # Collect
        for _ in range(n_steps):
            actions = []
            log_probs = []
            values = []

            for obs in obs_list:
                act, lp, val = agent.select_action(obs)
                actions.append(act)
                log_probs.append(lp)
                values.append(val)

            next_obs_list, rews, dones, infos = envs.step(actions)

            for i in range(num_envs):
                buffer.add(obs_list[i], [actions[i]], log_probs[i], rews[i], values[i], dones[i])
                total_steps += 1
                pbar.update(1)

                tb_cb.on_step(infos[i], total_steps)

                if dones[i] and 'episode' in infos[i]:
                    ep_count += 1
                    ep_r = infos[i]['episode']['r']
                    ep_buffer.append(ep_r)

                    logger.log_scalar('episode/reward', ep_r, ep_count)

                    if ep_count % 10 == 0:
                        avg = np.mean(ep_buffer[-100:]) if ep_buffer else 0
                        print(f"\nEp {ep_count} | Step {total_steps} | R: {ep_r:.2f} | Avg100: {avg:.2f}")

            obs_list = next_obs_list

        # Train
        last_vals = []
        for obs in obs_list:
            with torch.no_grad():
                _, _, lv = agent.select_action(obs)
                last_vals.append(lv)

        buffer.compute_returns_and_advantages(
            last_value=np.mean(last_vals),
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg['gae_lambda']
        )

        stats = agent.train(buffer, n_epochs=train_cfg['n_epochs'], batch_size=2048)

        # Update LR
        new_lr = lr_scheduler.step()
        tb_cb.writer.add_scalar('train/learning_rate', new_lr, total_steps)

        # Update entropy coef (schedule 0.01 → 0.001)
        progress = min(1.0, total_steps / train_cfg['total_timesteps'])
        new_ent = 0.01 - 0.009 * progress
        agent.ent_coef = new_ent
        tb_cb.writer.add_scalar('train/entropy_coef', new_ent, total_steps)

        tb_cb.on_train_step(stats, total_steps, agent=agent)
        logger.log_scalars('train', stats, total_steps)

        buffer.clear()

        # Eval
        eval_rew = eval_cb.evaluate(agent, total_steps, writer=tb_cb.writer)
        if eval_rew is not None:
            print(f"✓ Eval @ {total_steps}: {eval_rew:.2f}")

        # Save
        if total_steps % train_cfg['save_interval'] == 0 and total_steps > 0:
            ckpt = os.path.join(config['logging']['checkpoint_dir'], f'checkpoint_{total_steps}.pth')
            agent.save(ckpt)

            # Save VecNormalize stats
            if envs.normalizer:
                norm_path = os.path.join(config['logging']['checkpoint_dir'], f'vecnormalize_{total_steps}.pkl')
                envs.normalizer.save(norm_path)
                print(f"✓ Saved VecNormalize: {norm_path}")

    pbar.close()

    # Final save
    final = os.path.join(config['logging']['checkpoint_dir'], 'model_final.pth')
    agent.save(final)

    if envs.normalizer:
        envs.normalizer.save(os.path.join(config['logging']['checkpoint_dir'], 'vecnormalize_final.pkl'))

    envs.close()
    eval_env.close()
    logger.close()
    tb_cb.close()

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()

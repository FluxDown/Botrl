"""
🚀 SCRIPT FINAL avec VRAI MULTI-PROCESSING

✅ Multi-process (1 process par env) = VRAI parallélisme
✅ Batched forward (1 forward pour N obs au lieu de N forwards)
✅ get_value_batch() déterministe pour bootstrap
✅ VecNormalize (obs + returns)
✅ LR schedule (3e-4 → 5e-5 linéaire)
✅ Entropy schedule (0.01 → 0.001)
✅ termination séparé de truncation (Goal vs Timeout)
✅ CUDA optimisé (torch.compile, TF32, cuDNN)
✅ OMP/MKL threads=1 (évite contention CPU)
✅ TensorBoard enrichi (reward breakdown, clip_fraction, kl, lr, entropy)
✅ tick_skip corrigé (8 au lieu de 160)

ATTENDU : 8-10x plus rapide que train.py
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing

# CRITIQUE : 1 thread BLAS par process principal
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

torch.set_num_threads(1)

# Nos imports
from src.networks.actor_critic import ActorCritic
from src.agents.ppo import PPO, RolloutBuffer
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.custom_callbacks import CustomTBCallback, EvalCallback
from src.utils.parallel_envs_mp import ParallelEnvsMP
from src.utils.vec_wrapper import SimpleVecNormalize, LRScheduler
from src.utils.worker import create_env

def make_env_global(config):
    """
    Factory function GLOBALE pour créer des envs (pickable sur Windows).

    IMPORTANT : Doit être au niveau module (pas locale) pour être pickable.
    """
    return create_env(config)


def setup_cuda(device):
    """CUDA optimizations"""
    if device.type != 'cuda':
        return

    try:
        torch.set_float32_matmul_precision('high')
        print("✓ TF32 matmul")
    except:
        pass

    try:
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark")
    except:
        pass

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Memory: {mem_gb:.1f}GB")


def get_num_envs(config):
    """Auto-ajuste num_envs selon CPU"""
    requested = config['environment'].get('num_envs', 8)
    auto = config['environment'].get('auto_adjust_envs', True)

    if not auto:
        return requested

    cpu_count = multiprocessing.cpu_count()
    optimal = min(requested, max(1, int(cpu_count * 0.75)))

    if optimal != requested:
        print(f"⚠ num_envs auto-adjust: {requested} → {optimal} (CPU={cpu_count})")

    return optimal


def train():
    """TRAINING FINAL avec vrai multi-processing"""
    config = load_config('config.yaml')

    # Seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("🚀 ROCKET LEAGUE BOT - MULTI-PROCESS TRAINING (FINAL)")
    print(f"{'='*70}")
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

    # Multi-process envs (VRAI parallélisme)
    num_envs = get_num_envs(config)
    print(f"\n✓ Creating {num_envs} parallel PROCESSES...")

    # Utiliser la factory GLOBALE (pickable sur Windows)
    envs = ParallelEnvsMP(make_env_global, num_envs, config)

    # Env d'évaluation (single)
    print("✓ Creating eval environment...")
    eval_env = create_env(config)

    # VecNormalize
    normalizer = SimpleVecNormalize(gamma=config['training']['gamma'])
    print("✓ VecNormalize enabled")

    # Dimensions
    obs_size = envs.get_obs_space()
    act_size = envs.get_action_space()
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

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETERS:")
    print(f"  Batch: {train_cfg['batch_size']} ({num_envs} envs × {n_steps} steps)")
    print(f"  LR: {train_cfg['learning_rate']:.0e} → {train_cfg.get('final_lr', 5e-5):.0e}")
    print(f"  Entropy: 0.01 → 0.001")
    print(f"  tick_skip: {config['environment']['tick_skip']}")
    print(f"{'='*70}\n")

    # Reset
    obs_list = envs.reset()
    obs_array = np.array(obs_list)
    obs_array = normalizer.normalize_obs_array(obs_array)
    obs_list = list(obs_array)

    pbar = tqdm(total=train_cfg['total_timesteps'], desc="Training", ncols=100)

    while total_steps < train_cfg['total_timesteps']:
        # COLLECT (n_steps par env)
        for _ in range(n_steps):
            # BATCHED FORWARD (1 forward au lieu de num_envs forwards)
            actions, log_probs, values = agent.select_actions_batch(obs_list)

            # Step dans TOUS les envs (parallèle dans workers)
            next_obs_list, rews, dones, infos = envs.step(actions.tolist())

            # Normaliser
            next_obs_array = np.array(next_obs_list)
            next_obs_array = normalizer.normalize_obs_array(next_obs_array)

            rew_array = np.array(rews)
            done_array = np.array(dones)
            rew_array = normalizer.normalize_reward(rew_array, done_array)

            # Stocker dans buffer
            for i in range(num_envs):
                buffer.add(obs_list[i], [actions[i]], log_probs[i], rew_array[i], values[i], dones[i])
                total_steps += 1
                pbar.update(1)

                # Callback (toutes les 1k steps seulement)
                if total_steps % 1000 == 0:
                    tb_cb.on_step(infos[i], total_steps)

                # Episode terminé
                if dones[i] and 'episode' in infos[i]:
                    ep_count += 1
                    ep_r = infos[i]['episode']['r']
                    ep_buffer.append(ep_r)

                    logger.log_scalar('episode/reward', ep_r, ep_count)

                    if ep_count % 10 == 0:
                        avg = np.mean(ep_buffer[-100:]) if ep_buffer else 0
                        pbar.set_postfix({'ep': ep_count, 'r': f'{ep_r:.1f}', 'avg100': f'{avg:.1f}'})

            obs_list = list(next_obs_array)

        # TRAIN
        # Bootstrap value (BATCHED + déterministe)
        last_values = agent.get_value_batch(obs_list)

        buffer.compute_returns_and_advantages(
            last_value=np.mean(last_values),
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg['gae_lambda']
        )

        stats = agent.train(buffer, n_epochs=train_cfg['n_epochs'], batch_size=2048)

        # Update LR
        new_lr = lr_scheduler.step()
        tb_cb.writer.add_scalar('train/learning_rate', new_lr, total_steps)

        # Update entropy (0.01 → 0.001)
        progress = min(1.0, total_steps / train_cfg['total_timesteps'])
        new_ent = 0.01 - 0.009 * progress
        agent.ent_coef = new_ent
        tb_cb.writer.add_scalar('train/entropy_coef', new_ent, total_steps)

        # Log clip_fraction et KL (metriques PPO importantes)
        if 'clip_fraction' in stats:
            tb_cb.writer.add_scalar('train/clip_fraction', stats['clip_fraction'], total_steps)
        if 'kl_divergence' in stats:
            tb_cb.writer.add_scalar('train/kl_divergence', stats['kl_divergence'], total_steps)

        tb_cb.on_train_step(stats, total_steps, agent=agent)
        logger.log_scalars('train', stats, total_steps)

        buffer.clear()

        # Eval
        eval_rew = eval_cb.evaluate(agent, total_steps, writer=tb_cb.writer)
        if eval_rew is not None:
            pbar.write(f"✓ Eval @ {total_steps}: {eval_rew:.2f}")

        # Save
        if total_steps % train_cfg['save_interval'] == 0 and total_steps > 0:
            ckpt = os.path.join(config['logging']['checkpoint_dir'], f'checkpoint_{total_steps}.pth')
            agent.save(ckpt)

            # Save VecNormalize
            norm_path = os.path.join(config['logging']['checkpoint_dir'], f'vecnormalize_{total_steps}.pkl')
            normalizer.save(norm_path)
            pbar.write(f"✓ Saved checkpoint + vecnormalize @ {total_steps}")

    pbar.close()

    # Final save
    final = os.path.join(config['logging']['checkpoint_dir'], 'model_final.pth')
    agent.save(final)
    normalizer.save(os.path.join(config['logging']['checkpoint_dir'], 'vecnormalize_final.pkl'))

    envs.close()
    eval_env.close()
    logger.close()
    tb_cb.close()

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # IMPORTANT : multiprocessing sur Windows
    multiprocessing.set_start_method('spawn', force=True)
    train()

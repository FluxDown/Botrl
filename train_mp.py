# train_mp.py (extraits essentiels)

import os, time
import numpy as np
import torch
import multiprocessing as mp

from src.utils.config import load_config
from src.utils.parallel_envs_mp import ParallelEnvsMP
from src.utils.vec_wrapper import SimpleVecNormalize
from src.utils.worker import create_env
from src.utils.logger import Logger
from src.utils.custom_callbacks import CustomTBCallback, EvalCallback
from src.agents.ppo import PPO, RolloutBuffer
from src.networks.actor_critic import ActorCritic

def make_env_fn(config: dict, seed: int = 0):
    return create_env(config, seed=seed)

def train():
    config = load_config("config.yaml")

    # Device + perf flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # Dirs/loggers
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['tensorboard'],
        use_wandb=config['logging']['wandb'],
        wandb_config={'project': config['logging']['wandb_project'], 'config': config}
    )
    tb_cb = CustomTBCallback(config['logging']['log_dir'],
                             reward_keys=("goal","touch","progress","boost","demo","aerial"),
                             verbose=1)

    # Envs MP
    num_envs = config['environment'].get('num_envs', 8)
    base_seed = config['training'].get('seed', 42)
    envs = ParallelEnvsMP(config, make_env_fn, num_envs, base_seed=base_seed)
    eval_env = make_env_fn(config, seed=base_seed+999)

    # Reset + obs_dim
    obs_list = envs.reset()
    obs_dim = len(obs_list[0])
    act_size = 90  # LookupTableAction

    # Normalizer
    normalizer = SimpleVecNormalize(gamma=config['training']['gamma'], training=True)
    normalizer.init(num_envs=num_envs, obs_shape=(obs_dim,))
    # premier passage d’obs
    obs_batch = normalizer.normalize_obs_array(np.asarray(obs_list, dtype=np.float32))
    obs_list = [obs_batch[i] for i in range(num_envs)]

    # Policy / Agent
    net_cfg = config['network']
    policy = ActorCritic(
        obs_space_size=obs_dim,
        action_space_size=act_size,
        policy_layers=net_cfg['policy_layers'],
        value_layers=net_cfg['value_layers'],
        activation=net_cfg['activation'],
        continuous_actions=False
    )
    agent = PPO(
        policy_network=policy,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        vf_coef=config['training']['vf_coef'],
        ent_coef=config['training']['ent_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        device=device
    )
    # torch.compile (si CUDA)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            agent.policy = torch.compile(agent.policy, mode="reduce-overhead")
        except Exception:
            pass

    # Buffer
    batch_size = config['training']['batch_size']  # ex: 4096
    n_steps = config['training'].get('n_steps', batch_size // num_envs)  # ex: 512
    buffer = RolloutBuffer(buffer_size=batch_size, obs_dim=obs_dim, action_dim=1, device=device)

    # Eval callback
    eval_cb = EvalCallback(
        eval_env=eval_env,
        eval_freq=config['training'].get('eval_freq', 100000),
        n_eval_episodes=config['training'].get('n_eval_episodes', 10),
        best_model_path=os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth'),
        verbose=1
    )

    total_steps = 0
    pbar_total = config['training']['total_timesteps']
    print("\n" + "="*70)
    print(f"HYPERPARAMETERS:\n  Batch: {batch_size} ({num_envs} envs × {n_steps} steps)\n"
          f"  LR: {config['training']['learning_rate']:.0e} → {config['training'].get('final_lr',5e-5):.0e}\n"
          f"  Entropy: {config['training']['ent_coef']} → 0.001\n"
          f"  tick_skip: {config['environment'].get('tick_skip', 'NA')}")
    print("="*70)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=pbar_total, desc="Training")

        while total_steps < pbar_total:
            # -------- Collecte --------
            t0 = time.perf_counter()

            # ACTIONS (batch si possible)
            # Option 1: si tu as agent.select_action_batch(...)
            if hasattr(agent, "select_action_batch"):
                acts, logps, vals = agent.select_action_batch(np.asarray(obs_list, dtype=np.float32))
            else:
                # fallback: boucle
                acts, logps, vals = [], [], []
                for obs in obs_list:
                    a, lp, v = agent.select_action(obs)
                    acts.append(a); logps.append(lp); vals.append(v)

            next_obs_list, rews, dones, infos = envs.step(acts)

            # normalisation rewards/obs (array!)
            rew_array  = normalizer.normalize_reward(np.asarray(rews, dtype=np.float32),
                                                     np.asarray(dones, dtype=np.float32))
            obs_array  = normalizer.normalize_obs_array(np.asarray(next_obs_list, dtype=np.float32))
            next_obs_list = [obs_array[i] for i in range(num_envs)]

            # buffer fill
            for i in range(num_envs):
                buffer.add(obs_list[i], [acts[i]], logps[i], rew_array[i], vals[i], dones[i])
                total_steps += 1
                pbar.update(1)
                # TB metrics par step (léger)
                tb_cb.on_step(infos[i], total_steps)

            obs_list = next_obs_list
            # boucle collecte jusqu’à remplir n_steps
            for _ in range(n_steps-1):
                if hasattr(agent, "select_action_batch"):
                    acts, logps, vals = agent.select_action_batch(np.asarray(obs_list, dtype=np.float32))
                else:
                    acts, logps, vals = [], [], []
                    for obs in obs_list:
                        a, lp, v = agent.select_action(obs)
                        acts.append(a); logps.append(lp); vals.append(v)

                next_obs_list, rews, dones, infos = envs.step(acts)
                rew_array  = normalizer.normalize_reward(np.asarray(rews, dtype=np.float32),
                                                         np.asarray(dones, dtype=np.float32))
                obs_array  = normalizer.normalize_obs_array(np.asarray(next_obs_list, dtype=np.float32))
                next_obs_list = [obs_array[i] for i in range(num_envs)]

                for i in range(num_envs):
                    buffer.add(obs_list[i], [acts[i]], logps[i], rew_array[i], vals[i], dones[i])
                    total_steps += 1
                    pbar.update(1)
                    tb_cb.on_step(infos[i], total_steps)

                obs_list = next_obs_list

            t1 = time.perf_counter()

            # -------- Update PPO --------
            # bootstrap value moyen
            last_vals = []
            for obs in obs_list:
                with torch.no_grad():
                    _, _, v = agent.select_action(obs)  # remplace par value(obs) si tu l’as
                    last_vals.append(v)

            buffer.compute_returns_and_advantages(
                last_value=float(np.mean(last_vals)),
                gamma=config['training']['gamma'],
                gae_lambda=config['training']['gae_lambda']
            )

            stats = agent.train(buffer, n_epochs=config['training']['n_epochs'], batch_size=2048)
            buffer.clear()
            t2 = time.perf_counter()

            # Logs timing
            print(f"collecte: {t1-t0:.2f}s | update: {t2-t1:.2f}s")

            # LR & entropy schedules (existant)
            # ... ton code de scheduler LR + entropy ici ...
            tb_cb.on_train_step(stats, total_steps, agent=agent)

            # Eval (pas trop souvent)
            if (total_steps // (num_envs)) % config['training'].get('eval_freq', 100000) == 0:
                eval_rew = eval_cb.evaluate(agent, total_steps, writer=tb_cb.writer)
                if eval_rew is not None:
                    print(f"✓ Eval @ {total_steps}: {eval_rew:.2f}")

            # Save checkpoint régulier
            save_interval = config['training'].get('save_interval', 100000)
            if total_steps % save_interval == 0 and total_steps > 0:
                ckpt_path = os.path.join(config['logging']['checkpoint_dir'], f'checkpoint_{total_steps}.pth')
                norm_path = os.path.join(config['logging']['checkpoint_dir'], f'vecnormalize_{total_steps}.npz')
                agent.save(ckpt_path)
                normalizer.save(norm_path)
                print(f"\n✓ Saved checkpoint @ {total_steps}")

            # Save LIVE policy pour RLBot (toutes les 4k steps = chaque update PPO)
            if total_steps % 4096 == 0 and total_steps > 0:
                live_path_tmp = os.path.join(config['logging']['checkpoint_dir'], 'latest_policy.tmp')
                live_path = os.path.join(config['logging']['checkpoint_dir'], 'latest_policy.pt')
                torch.save({
                    'policy_state_dict': agent.policy.state_dict(),
                    'obs_mean': normalizer.obs_rms.mean if normalizer.obs_rms else None,
                    'obs_var': normalizer.obs_rms.var if normalizer.obs_rms else None,
                    'total_steps': total_steps
                }, live_path_tmp)
                os.replace(live_path_tmp, live_path)  # Atomic swap
                print(f"\n[LIVE] Policy saved @ {total_steps} steps")

        pbar.close()

    except KeyboardInterrupt:
        print("\n⚠ Interruption utilisateur: sauvegarde et fermeture propre...")
        try:
            agent.save(os.path.join(config['logging']['checkpoint_dir'], 'interrupt_checkpoint.pth'))
            normalizer.save(os.path.join(config['logging']['checkpoint_dir'], 'vecnormalize_interrupt.npz'))
        except Exception as e:
            print(f"(!) Sauvegarde interrompue: {e}")
    finally:
        try: envs.close()
        except Exception as e: print(f"close envs: {e}")
        try: eval_env.close()
        except Exception as e: print(f"close eval_env: {e}")
        try: logger.close()
        except Exception as e: print(f"close logger: {e}")
        try: tb_cb.close()
        except Exception as e: print(f"close tb: {e}")


if __name__ == "__main__":
    # Windows spawn + threads BLAS
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    torch.set_num_threads(1)

    train()

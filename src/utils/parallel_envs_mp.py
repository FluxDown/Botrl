"""
VRAI multi-processing pour environnements parallèles.
1 process Python par env = pas de GIL = vrai parallélisme.

Basé sur le pattern SubprocVecEnv de Stable-Baselines3.
"""

from multiprocessing import Process, Pipe
import numpy as np


def _worker(env_fn, conn, worker_id, config):
    """
    Worker process qui tourne en parallèle.

    Args:
        env_fn: Factory function qui retourne un env
        conn: Pipe de communication avec le process principal
        worker_id: ID du worker (pour debug)
        config: Configuration dict (passée explicitement pour Windows)
    """
    try:
        env = env_fn(config)

        # Reset initial
        result = env.reset()
        obs_dict = result[0] if isinstance(result, tuple) else result

        # En 1v1 : 2 agents, récupérer leurs IDs réels
        if isinstance(obs_dict, dict):
            agent_ids = list(obs_dict.keys())
            agent_id = agent_ids[0]  # Notre agent (premier)
            other_agent_id = agent_ids[1] if len(agent_ids) > 1 else None
            obs = obs_dict[agent_id]
            print(f"[Worker {worker_id}] Controlling agent {agent_id}, opponent {other_agent_id}")
        else:
            # Cas single agent
            agent_id = 0
            other_agent_id = None
            obs = obs_dict

        # Episode tracking
        ep_reward = 0.0
        ep_length = 0

        # Boucle de communication
        while True:
            cmd, data = conn.recv()

            if cmd == "step":
                act = int(data)

                # Action dict : notre agent + adversaire noop
                action_dict = {agent_id: np.array([act])}
                if other_agent_id is not None:
                    action_dict[other_agent_id] = np.array([0])  # Adversaire = noop

                # Step
                step_result = env.step(action_dict)

                if len(step_result) == 5:
                    obs_dict, rew_dict, term_dict, trunc_dict, info = step_result
                else:
                    obs_dict, rew_dict, term_dict, trunc_dict = step_result
                    info = {}

                # Extraire pour notre agent
                obs = obs_dict[agent_id]
                reward = float(rew_dict[agent_id])
                terminated = bool(term_dict[agent_id])
                truncated = bool(trunc_dict[agent_id])
                done = terminated or truncated

                ep_reward += reward
                ep_length += 1

                # Propager reward_parts si disponible
                if hasattr(env, 'reward_fn') and hasattr(env.reward_fn, '_last_reward_parts'):
                    info['reward_parts'] = env.reward_fn._last_reward_parts

                # Si done, ajouter episode info et reset
                if done:
                    info['episode'] = {'r': ep_reward, 'l': ep_length}
                    info['success'] = 1.0 if ep_reward > 50 else 0.0

                    # Reset
                    result = env.reset()
                    obs_dict = result[0] if isinstance(result, tuple) else result
                    obs = obs_dict[agent_id] if isinstance(obs_dict, dict) else obs_dict

                    ep_reward = 0.0
                    ep_length = 0

                # Envoyer résultat
                conn.send((obs, reward, done, info))

            elif cmd == "get_obs":
                conn.send(obs)

            elif cmd == "close":
                env.close()
                conn.close()
                break

    except Exception as e:
        print(f"Worker {worker_id} crashed: {e}")
        import traceback
        traceback.print_exc()
        conn.close()


class ParallelEnvsMP:
    """
    Gestionnaire d'environnements multi-process VRAIS.

    Chaque env tourne dans son propre process Python.
    Communication via Pipes (non-bloquante).

    Usage:
        def make_env(config):
            return create_env(config)

        envs = ParallelEnvsMP(make_env, num_envs=8, config=config)
        obs_list = envs.reset()
        obs_list, rewards, dones, infos = envs.step(actions)
        envs.close()
    """

    def __init__(self, make_env, num_envs, config):
        """
        Args:
            make_env: Factory function (config) -> RLGym env
            num_envs: Nombre d'environnements parallèles
            config: Configuration dict (sera passée aux workers)
        """
        self.num_envs = num_envs
        self.parents = []
        self.procs = []

        print(f"Creating {num_envs} parallel processes...")

        # Créer les workers (passer config explicitement)
        for i in range(num_envs):
            parent_conn, child_conn = Pipe()
            proc = Process(target=_worker, args=(make_env, child_conn, i, config), daemon=True)
            proc.start()

            self.parents.append(parent_conn)
            self.procs.append(proc)

        # Récupérer les observations initiales
        for conn in self.parents:
            conn.send(("get_obs", None))

        self.obs = [conn.recv() for conn in self.parents]

        print(f"✓ {num_envs} processes started")

    def reset(self):
        """
        Reset tous les environnements.

        Returns:
            obs_list: Liste de num_envs observations
        """
        for conn in self.parents:
            conn.send(("get_obs", None))

        self.obs = [conn.recv() for conn in self.parents]
        return self.obs

    def step(self, actions):
        """
        Exécute les actions dans TOUS les envs EN PARALLÈLE.

        Args:
            actions: Liste ou array de num_envs actions (int)

        Returns:
            obs_list: Liste de num_envs observations
            rewards: Liste de num_envs rewards (float)
            dones: Liste de num_envs dones (bool)
            infos: Liste de num_envs info dicts
        """
        # Envoyer les actions à tous les workers (NON-BLOQUANT)
        for conn, action in zip(self.parents, actions):
            conn.send(("step", action))

        # Recevoir les résultats (les workers tournent en PARALLÈLE)
        results = [conn.recv() for conn in self.parents]

        # Unpack
        obs, rewards, dones, infos = zip(*results)

        self.obs = list(obs)

        return list(obs), list(rewards), list(dones), list(infos)

    def close(self):
        """Ferme tous les workers"""
        for conn in self.parents:
            try:
                conn.send(("close", None))
            except:
                pass

        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def get_obs_space(self):
        """Retourne la dimension de l'espace d'observation"""
        return len(self.obs[0])

    def get_action_space(self):
        """Retourne le nombre d'actions (LookupTable = 90)"""
        return 90

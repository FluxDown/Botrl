# src/utils/parallel_envs_mp.py
from multiprocessing import Process, Pipe
import numpy as np


def _build_action_dict(action_in, agent_ids, controlled_idx=0):
    """
    action_in:
      - int/np.int (action unique pour l'agent contrôlé) -> autres agents = 0 (no-op)
      - list/tuple/ndarray de len == n_agents -> actions par agent.
    """
    n_agents = len(agent_ids)

    # liste complète fournie ?
    if isinstance(action_in, (list, tuple, np.ndarray)):
        if len(action_in) != n_agents:
            raise KeyError(f"Expected actions for {n_agents} agents but received {len(action_in)}.")
        return {agent_ids[i]: np.array([int(action_in[i])]) for i in range(n_agents)}

    # scalaire -> on contrôle seulement controlled_idx, le reste = 0
    a = int(action_in)
    act = {}
    for i, aid in enumerate(agent_ids):
        act[aid] = np.array([a if i == controlled_idx else 0], dtype=np.int64)
    return act


def _unpack_reset(reset_out):
    """
    Supporte reset() qui renvoie (obs_dict,) ou (obs_dict, info) ou obs_dict.
    Retourne seulement obs_dict.
    """
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _unpack_step(result, agent_ids, controlled_idx):
    """
    Supporte step() qui renvoie:
      - 5 valeurs: (obs_dict, rew_dict, term_dict, trunc_dict, info)
      - 4 valeurs: (obs_dict, rew_dict, done_dict, info)
    Retourne: (obs_ctrl, rew_ctrl, done_bool, info, obs_dict_full)
    """
    if not isinstance(result, tuple):
        raise RuntimeError(f"Env.step returned unexpected type: {type(result)}")

    if len(result) == 5:
        obs_d, rew_d, term_d, trunc_d, info = result
        term = bool(list(term_d.values())[0])
        trunc = bool(list(trunc_d.values())[0])
        done = term or trunc
        obs_ctrl = obs_d[agent_ids[controlled_idx]]
        rew_ctrl = float(rew_d[agent_ids[controlled_idx]])
        return obs_ctrl, rew_ctrl, done, info, obs_d

    if len(result) == 4:
        obs_d, rew_d, done_d, info = result
        done = bool(list(done_d.values())[0])
        obs_ctrl = obs_d[agent_ids[controlled_idx]]
        rew_ctrl = float(rew_d[agent_ids[controlled_idx]])
        return obs_ctrl, rew_ctrl, done, info, obs_d

    raise ValueError(f"Env.step returned {len(result)} values (expected 4 or 5).")


def _is_error_msg(msg):
    """Vérifie si msg est du type ("__error__", "...") sans provoquer d'ambiguïté NumPy."""
    return (
        isinstance(msg, tuple)
        and len(msg) == 2
        and isinstance(msg[0], str)
        and msg[0] == "__error__"
    )


def _worker(config, make_env_fn, conn, rank=0, base_seed=0):
    """
    Worker multiprocess Windows-safe, multi-agent aware.
    - Détecte les agent_ids au reset et le nombre d'agents.
    - Si le parent envoie UN scalaire, on le joue pour l'agent contrôlé, autres = no-op.
    - Si le parent envoie une liste d'actions == n_agents, on les applique 1:1.
    - Gère les 2 signatures de step() (4 ou 5 retours).
    """
    env = None
    try:
        env = make_env_fn(config, seed=base_seed + rank)

        # --- reset initial ---
        obs_dict = _unpack_reset(env.reset())
        if not isinstance(obs_dict, dict) or len(obs_dict) == 0:
            raise RuntimeError("Env reset did not return a dict of observations per agent.")
        agent_ids = list(obs_dict.keys())
        controlled_idx = 0  # on contrôle le premier agent (change si besoin)
        obs = obs_dict[agent_ids[controlled_idx]]

        # --- warm-up RocketSim ---
        try:
            for _ in range(10):
                act_dict = _build_action_dict(0, agent_ids, controlled_idx=controlled_idx)
                step_out = env.step(act_dict)
                _, _, done_tmp, _, _ = _unpack_step(step_out, agent_ids, controlled_idx)
                if done_tmp:
                    _ = env.reset()
        except Exception:
            pass

        # --- obs propre après warm-up ---
        obs_dict = _unpack_reset(env.reset())
        agent_ids = list(obs_dict.keys())  # revalide
        obs = obs_dict[agent_ids[controlled_idx]]

        # >>> ENVOI OBS INITIALE <<<
        conn.send(obs)

        # --- boucle principale ---
        while True:
            if not conn.poll(0.01):  # 10 ms
                continue

            cmd, data = conn.recv()

            if cmd == "step":
                try:
                    act_dict = _build_action_dict(data, agent_ids, controlled_idx=controlled_idx)
                except KeyError as e:
                    conn.send(("__error__", repr(e)))
                    continue

                step_out = env.step(act_dict)
                obs, rew, done, info, obs_full = _unpack_step(step_out, agent_ids, controlled_idx)

                if done:
                    obs_dict = _unpack_reset(env.reset())
                    agent_ids = list(obs_dict.keys())  # revalide au cas où
                    obs = obs_dict[agent_ids[controlled_idx]]

                conn.send((obs, rew, done, info))

            elif cmd == "get_obs":
                conn.send(obs)

            elif cmd == "close":
                try: env.close()
                except Exception: pass
                try: conn.close()
                except Exception: pass
                break

    except KeyboardInterrupt:
        try: env and env.close()
        except Exception: pass
        try: conn.close()
        except Exception: pass

    except Exception as e:
        try: conn.send(("__error__", repr(e)))
        except Exception: pass
        try: conn.close()
        except Exception: pass


class ParallelEnvsMP:
    """
    Contrôleur de N workers.
    - Reçoit l'obs initiale poussée par chaque worker.
    - step(actions): accepte liste d'entiers (un par env) OU liste de listes (par agent).
    - Remonte proprement les erreurs ("__error__", "...").
    """
    def __init__(self, config, make_env_fn, n, base_seed=0):
        self.n = int(n)
        self.parents, self.procs = [], []

        for i in range(self.n):
            p_conn, c_conn = Pipe()
            p = Process(
                target=_worker,
                args=(config, make_env_fn, c_conn, i, base_seed),
                daemon=True
            )
            p.start()
            self.parents.append(p_conn)
            self.procs.append(p)

        # Réception des obs initiales (ou d'une erreur)
        self.obs = []
        for c in self.parents:
            msg = c.recv()
            if _is_error_msg(msg):
                self.close()
                raise RuntimeError(f"Worker error at init: {msg[1]}")
            self.obs.append(msg)

    def reset(self):
        for c in self.parents:
            try: c.send(("get_obs", None))
            except Exception: pass
        self.obs = [c.recv() for c in self.parents]
        return self.obs

    def step(self, actions):
        # actions peut être:
        #  - liste d'entiers (longueur = n_envs)
        #  - liste de listes (longueur = n_envs, chaque sous-liste = n_agents)
        for c, a in zip(self.parents, actions):
            try: c.send(("step", a))
            except Exception: pass

        results, remaining = [], list(self.parents)
        while remaining:
            for c in remaining[:]:
                try:
                    if c.poll(0.01):  # 10 ms
                        results.append(c.recv())
                        remaining.remove(c)
                except (EOFError, OSError):
                    results.append(("__error__", "Pipe closed"))
                    remaining.remove(c)

        for r in results:
            if _is_error_msg(r):
                self.close()
                raise RuntimeError(f"Worker error during step: {r[1]}")

        obs, rew, done, infos = zip(*results)
        self.obs = list(obs)
        return list(obs), list(rew), list(done), list(infos)

    def close(self):
        for c in self.parents:
            try: c.send(("close", None))
            except Exception: pass
        for p in self.procs:
            try: p.join(timeout=2.0)
            except Exception: pass
        for p in self.procs:
            if p.is_alive():
                try: p.terminate()
                except Exception: pass

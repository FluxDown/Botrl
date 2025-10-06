"""
Contrôleur multi-process avec VRAI parallélisme
"""

import numpy as np
from multiprocessing import Process, Pipe


class ParallelEnvsMP:
    """
    Gestionnaire d'environnements multi-process VRAIS.
    Chaque env tourne dans son propre process Python.
    """

    def __init__(self, config, num_envs):
        """
        Args:
            config: Configuration dict
            num_envs: Nombre d'environnements parallèles
        """
        from src.utils.worker import worker

        self.num_envs = num_envs
        self.procs = []
        self.parent_conns = []

        print(f"Creating {num_envs} parallel processes...")

        # Créer les workers (1 process par env)
        for i in range(num_envs):
            parent_conn, child_conn = Pipe()
            proc = Process(target=worker, args=(i, config, child_conn), daemon=True)
            proc.start()

            self.procs.append(proc)
            self.parent_conns.append(parent_conn)

        # Récupérer les observations initiales
        for conn in self.parent_conns:
            conn.send(("get_obs", None))

        self.current_obs = [conn.recv() for conn in self.parent_conns]

        print(f"✓ {num_envs} processes started")

    def step(self, actions):
        """
        Exécute les actions dans TOUS les envs en PARALLÈLE.

        Args:
            actions: Liste de num_envs actions (int)

        Returns:
            observations, rewards, dones, infos
        """
        # Envoyer les actions à tous les workers EN MÊME TEMPS
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))

        # Recevoir les résultats (les workers tournent en parallèle)
        results = [conn.recv() for conn in self.parent_conns]

        # Unpack
        obs, rewards, dones, infos = zip(*results)

        self.current_obs = list(obs)

        return list(obs), list(rewards), list(dones), list(infos)

    def reset(self):
        """Reset tous les envs"""
        for conn in self.parent_conns:
            conn.send(("get_obs", None))

        self.current_obs = [conn.recv() for conn in self.parent_conns]
        return self.current_obs

    def close(self):
        """Ferme tous les workers"""
        for conn in self.parent_conns:
            conn.send(("close", None))

        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def get_obs_space(self):
        """Retourne la dimension de l'espace d'observation"""
        return len(self.current_obs[0])

    def get_action_space(self):
        """Retourne le nombre d'actions (LookupTable = 90)"""
        return 90

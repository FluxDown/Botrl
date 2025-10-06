import numpy as np
from rlgym.api import ObsBuilder
from rlgym.rocket_league.api import GameState


class AdvancedObs(ObsBuilder):
    """
    Observation builder avancé qui fournit des informations riches au réseau de neurones:
    - Position et vitesse de la voiture (relative et absolue)
    - Rotation de la voiture (matrice de rotation aplatie)
    - Boost disponible
    - Position et vitesse de la balle (relative et absolue)
    - Distance et direction vers la balle
    - Positions des coéquipiers et adversaires
    - Distance aux buts
    """

    def __init__(self, team_size=1, tick_skip=8):
        super().__init__()
        self.team_size = team_size
        self.tick_skip = tick_skip

    def get_obs_space(self) -> int:
        """
        Calcule la taille de l'espace d'observation

        Breakdown:
        - Voiture: 3 (pos) + 3 (vel) + 3 (ang_vel) + 9 (rotation matrix) + 1 (boost) + 1 (on_ground) + 1 (has_flip) = 21
        - Balle: 3 (pos) + 3 (vel) + 3 (ang_vel) = 9
        - Balle relative: 3 (pos) + 3 (vel) = 6
        - Buts: 2 (distances) = 2
        - Autres joueurs: (team_size-1 + team_size) * 7 (pos + vel + on_ground) = (2*team_size-1)*7
        """
        base_size = 21 + 9 + 6 + 2  # 38
        other_players = (2 * self.team_size - 1) * 7
        return base_size + other_players

    def reset(self, agents: list[int], initial_state: GameState, shared_info: dict = None) -> None:
        """Reset l'observation builder"""
        pass

    def build_obs(self, agents: list[int], state: GameState, shared_info: dict) -> dict:
        """Construit les observations pour chaque agent"""
        obs = {}

        for agent_id in agents:
            car = state.cars[agent_id]
            obs[agent_id] = self._build_single_obs(agent_id, car, state)

        return obs

    def _build_single_obs(self, agent_id: int, car, state: GameState) -> np.ndarray:
        """Construit l'observation pour un seul agent"""
        obs_list = []

        # === Informations sur la voiture ===
        # Position (normalisée)
        car_pos = np.array(car.physics.position)
        obs_list.extend(car_pos / 4096)  # Normaliser par la taille du terrain

        # Vélocité linéaire (normalisée)
        car_vel = np.array(car.physics.linear_velocity)
        obs_list.extend(car_vel / 2300)  # Normaliser par vitesse max

        # Vélocité angulaire (normalisée)
        car_ang_vel = np.array(car.physics.angular_velocity)
        obs_list.extend(car_ang_vel / 5.5)  # Normaliser

        # Matrice de rotation (orientation de la voiture)
        rotation_mtx = np.array(car.physics.rotation_mtx)
        # rotation_mtx est une matrice 3x3, on l'aplatit
        if rotation_mtx.shape == (3, 3):
            obs_list.extend(rotation_mtx.flatten())
        else:
            # Si c'est déjà un vecteur aplati
            obs_list.extend(rotation_mtx[:9])

        # Boost (normalisé)
        obs_list.append(car.boost_amount / 100)

        # Au sol
        obs_list.append(1.0 if car.on_ground else 0.0)

        # A le flip disponible
        obs_list.append(1.0 if car.has_flip else 0.0)

        # === Informations sur la balle ===
        # Position absolue (normalisée)
        ball_pos = np.array(state.ball.position)
        obs_list.extend(ball_pos / 4096)

        # Vélocité (normalisée)
        ball_vel = np.array(state.ball.linear_velocity)
        obs_list.extend(ball_vel / 6000)

        # Vélocité angulaire (normalisée)
        ball_ang_vel = np.array(state.ball.angular_velocity)
        obs_list.extend(ball_ang_vel / 6)

        # === Informations relatives à la balle ===
        # Position relative à la voiture
        ball_relative_pos = ball_pos - car_pos
        obs_list.extend(ball_relative_pos / 10000)

        # Vélocité relative
        ball_relative_vel = ball_vel - car_vel
        obs_list.extend(ball_relative_vel / 6000)

        # === Distances aux buts ===
        # Distance au but adverse
        if car.team_num == 0:  # Équipe bleue
            enemy_goal = np.array([0, 5120, 0])
            own_goal = np.array([0, -5120, 0])
        else:  # Équipe orange
            enemy_goal = np.array([0, -5120, 0])
            own_goal = np.array([0, 5120, 0])

        dist_to_enemy_goal = np.linalg.norm(car_pos - enemy_goal) / 10000
        dist_to_own_goal = np.linalg.norm(car_pos - own_goal) / 10000
        obs_list.extend([dist_to_enemy_goal, dist_to_own_goal])

        # === Informations sur les autres joueurs ===
        # Trier les voitures par équipe
        teammates = []
        opponents = []

        for other_id, other_car in state.cars.items():
            if other_id == agent_id:
                continue

            if other_car.team_num == car.team_num:
                teammates.append(other_car)
            else:
                opponents.append(other_car)

        # Ajouter les coéquipiers (jusqu'à team_size - 1)
        for i in range(self.team_size - 1):
            if i < len(teammates):
                teammate = teammates[i]
                teammate_pos = np.array(teammate.physics.position)
                teammate_vel = np.array(teammate.physics.linear_velocity)

                obs_list.extend((teammate_pos - car_pos) / 10000)
                obs_list.extend(teammate_vel / 2300)
                obs_list.append(1.0 if teammate.on_ground else 0.0)
            else:
                # Padding si pas assez de coéquipiers
                obs_list.extend([0] * 7)

        # Ajouter les adversaires (jusqu'à team_size)
        for i in range(self.team_size):
            if i < len(opponents):
                opponent = opponents[i]
                opponent_pos = np.array(opponent.physics.position)
                opponent_vel = np.array(opponent.physics.linear_velocity)

                obs_list.extend((opponent_pos - car_pos) / 10000)
                obs_list.extend(opponent_vel / 2300)
                obs_list.append(1.0 if opponent.on_ground else 0.0)
            else:
                # Padding si pas assez d'adversaires
                obs_list.extend([0] * 7)

        return np.array(obs_list, dtype=np.float32)

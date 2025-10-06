import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.api import GameState


class CombinedReward(RewardFunction):
    """
    Fonction de récompense complexe combinant plusieurs objectifs:
    - Marquer des buts
    - Toucher la balle
    - Se diriger vers la balle
    - Diriger la balle vers le but adverse
    - Gérer le boost efficacement
    - Réaliser des aériens
    - Faire des démos
    """

    def __init__(
        self,
        goal_weight=10.0,
        concede_weight=-10.0,
        touch_ball_weight=0.5,
        velocity_player_to_ball_weight=0.3,
        velocity_ball_to_goal_weight=0.5,
        save_boost_weight=0.1,
        demo_weight=2.0,
        got_demoed_weight=-2.0,
        align_ball_goal_weight=0.3,
        aerial_weight=0.5,
        distance_to_ball_weight=-0.1
    ):
        super().__init__()
        self.goal_weight = goal_weight
        self.concede_weight = concede_weight
        self.touch_ball_weight = touch_ball_weight
        self.velocity_player_to_ball_weight = velocity_player_to_ball_weight
        self.velocity_ball_to_goal_weight = velocity_ball_to_goal_weight
        self.save_boost_weight = save_boost_weight
        self.demo_weight = demo_weight
        self.got_demoed_weight = got_demoed_weight
        self.align_ball_goal_weight = align_ball_goal_weight
        self.aerial_weight = aerial_weight
        self.distance_to_ball_weight = distance_to_ball_weight

        # État précédent
        self.last_ball_distance = {}  # Pour détecter les touches de balle
        self.last_had_car_contact = {}  # Pour détecter les bumps/demos

    def reset(self, agents: list[int], initial_state: GameState, shared_info: dict = None) -> None:
        """Reset l'état pour les agents"""
        for agent_id in agents:
            self.last_ball_distance[agent_id] = 10000  # Valeur initiale grande
            self.last_had_car_contact[agent_id] = False

    def get_rewards(self, agents: list[int], state: GameState, is_terminated: dict, is_truncated: dict, shared_info: dict = None) -> dict:
        """Calcule les récompenses pour chaque agent"""
        rewards = {}

        for agent_id in agents:
            reward = 0.0
            car = state.cars[agent_id]

            # Position et vitesse de la voiture
            car_pos = np.array(car.physics.position)
            car_vel = np.array(car.physics.linear_velocity)

            # Position et vitesse de la balle
            ball_pos = np.array(state.ball.position)
            ball_vel = np.array(state.ball.linear_velocity)

            # Déterminer le but adverse (basé sur l'équipe)
            if car.team_num == 0:  # Équipe bleue
                goal_pos = np.array([0, 5120, 0])  # But orange
                own_goal_pos = np.array([0, -5120, 0])  # But bleu
            else:  # Équipe orange
                goal_pos = np.array([0, -5120, 0])  # But bleu
                own_goal_pos = np.array([0, 5120, 0])  # But orange

            # 1. Récompense pour toucher la balle
            if car.ball_touches > 0:
                reward += self.touch_ball_weight

            # 2. Récompense pour se diriger vers la balle
            direction_to_ball = ball_pos - car_pos
            dist_to_ball = np.linalg.norm(direction_to_ball)

            if dist_to_ball > 0:
                direction_to_ball_norm = direction_to_ball / dist_to_ball
                velocity_to_ball = np.dot(car_vel, direction_to_ball_norm)
                reward += self.velocity_player_to_ball_weight * velocity_to_ball / 2300  # Normaliser par vitesse max

            # 3. Récompense pour diriger la balle vers le but
            direction_ball_to_goal = goal_pos - ball_pos
            dist_ball_to_goal = np.linalg.norm(direction_ball_to_goal)

            if dist_ball_to_goal > 0:
                direction_ball_to_goal_norm = direction_ball_to_goal / dist_ball_to_goal
                velocity_ball_to_goal = np.dot(ball_vel, direction_ball_to_goal_norm)
                reward += self.velocity_ball_to_goal_weight * velocity_ball_to_goal / 6000  # Normaliser

            # 4. Récompense pour économiser le boost
            boost_amount = car.boost_amount
            reward += self.save_boost_weight * boost_amount / 100

            # 5. Récompense pour aligner la balle avec le but
            if dist_to_ball < 500:  # Seulement si proche de la balle
                car_to_ball = ball_pos - car_pos
                ball_to_goal = goal_pos - ball_pos

                if np.linalg.norm(car_to_ball) > 0 and np.linalg.norm(ball_to_goal) > 0:
                    alignment = np.dot(
                        car_to_ball / np.linalg.norm(car_to_ball),
                        ball_to_goal / np.linalg.norm(ball_to_goal)
                    )
                    reward += self.align_ball_goal_weight * alignment

            # 6. Récompense pour les aériens
            if car_pos[2] > 300 and dist_to_ball < 400:  # En l'air et proche de la balle
                reward += self.aerial_weight

            # 7. Récompense pour bump/demo
            if car.had_car_contact and not self.last_had_car_contact.get(agent_id, False):
                reward += self.demo_weight
            self.last_had_car_contact[agent_id] = car.had_car_contact

            # 8. Pénalité pour se faire démo
            if car.is_demoed:
                reward += self.got_demoed_weight

            # 9. Pénalité pour être loin de la balle (encourage l'action)
            reward += self.distance_to_ball_weight * (dist_to_ball / 10000)

            rewards[agent_id] = reward

        return rewards

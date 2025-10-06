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

        # Pour propagation vers info (utilisé par RewardPartsToInfo wrapper)
        self._last_reward_parts = {}

    def reset(self, agents: list[int], initial_state: GameState, shared_info: dict = None) -> None:
        """Reset l'état pour les agents"""
        for agent_id in agents:
            self.last_ball_distance[agent_id] = 10000  # Valeur initiale grande
            self.last_had_car_contact[agent_id] = False

    def get_rewards(self, agents: list[int], state: GameState, is_terminated: dict, is_truncated: dict, shared_info: dict = None) -> dict:
        """Calcule les récompenses pour chaque agent"""
        rewards = {}

        # Initialiser shared_info pour stocker les reward_parts
        if shared_info is None:
            shared_info = {}
        if "reward_parts" not in shared_info:
            shared_info["reward_parts"] = {}

        for agent_id in agents:
            reward = 0.0
            reward_parts = {
                "goal": 0.0,
                "touch": 0.0,
                "progress": 0.0,
                "boost": 0.0,
                "demo": 0.0,
                "aerial": 0.0
            }

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
            touch_reward = 0.0
            if car.ball_touches > 0:
                touch_reward = self.touch_ball_weight
                reward += touch_reward
            reward_parts["touch"] = touch_reward

            # 2. Récompense pour se diriger vers la balle
            direction_to_ball = ball_pos - car_pos
            dist_to_ball = np.linalg.norm(direction_to_ball)

            progress_reward = 0.0
            if dist_to_ball > 0:
                direction_to_ball_norm = direction_to_ball / dist_to_ball
                velocity_to_ball = np.dot(car_vel, direction_to_ball_norm)
                progress_reward += self.velocity_player_to_ball_weight * velocity_to_ball / 2300  # Normaliser par vitesse max

            # 3. Récompense pour diriger la balle vers le but
            direction_ball_to_goal = goal_pos - ball_pos
            dist_ball_to_goal = np.linalg.norm(direction_ball_to_goal)

            if dist_ball_to_goal > 0:
                direction_ball_to_goal_norm = direction_ball_to_goal / dist_ball_to_goal
                velocity_ball_to_goal = np.dot(ball_vel, direction_ball_to_goal_norm)
                progress_reward += self.velocity_ball_to_goal_weight * velocity_ball_to_goal / 6000  # Normaliser

            reward += progress_reward
            reward_parts["progress"] = progress_reward

            # 4. Récompense pour économiser le boost
            boost_amount = car.boost_amount
            boost_reward = self.save_boost_weight * boost_amount / 100
            reward += boost_reward
            reward_parts["boost"] = boost_reward

            # 5. Récompense pour aligner la balle avec le but
            if dist_to_ball < 500:  # Seulement si proche de la balle
                car_to_ball = ball_pos - car_pos
                ball_to_goal = goal_pos - ball_pos

                if np.linalg.norm(car_to_ball) > 0 and np.linalg.norm(ball_to_goal) > 0:
                    alignment = np.dot(
                        car_to_ball / np.linalg.norm(car_to_ball),
                        ball_to_goal / np.linalg.norm(ball_to_goal)
                    )
                    alignment_reward = self.align_ball_goal_weight * alignment
                    reward += alignment_reward
                    reward_parts["progress"] += alignment_reward

            # 6. Récompense pour les aériens
            aerial_reward = 0.0
            if car_pos[2] > 300 and dist_to_ball < 400:  # En l'air et proche de la balle
                aerial_reward = self.aerial_weight
                reward += aerial_reward
            reward_parts["aerial"] = aerial_reward

            # 7. Récompense pour bump/demo
            demo_reward = 0.0
            if car.had_car_contact and not self.last_had_car_contact.get(agent_id, False):
                demo_reward = self.demo_weight
                reward += demo_reward
            self.last_had_car_contact[agent_id] = car.had_car_contact
            reward_parts["demo"] = demo_reward

            # 8. Pénalité pour se faire démo
            if car.is_demoed:
                got_demoed_reward = self.got_demoed_weight
                reward += got_demoed_reward
                reward_parts["demo"] += got_demoed_reward

            # 9. Pénalité pour être loin de la balle (encourage l'action)
            distance_penalty = self.distance_to_ball_weight * (dist_to_ball / 10000)
            reward += distance_penalty
            reward_parts["progress"] += distance_penalty

            # Stocker les reward_parts pour le logging
            shared_info["reward_parts"][agent_id] = reward_parts

            # NOUVEAU : Stocker aussi dans _last_reward_parts pour le wrapper
            self._last_reward_parts = reward_parts

            rewards[agent_id] = reward

        return rewards

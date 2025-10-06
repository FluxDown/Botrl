"""
Worker multi-process pour vrai parallélisme d'environnements
"""

import os
import numpy as np

# CRITIQUE : 1 thread BLAS par process (évite contention)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league.sim import RocketSimEngine

from src.obs.advanced_obs import AdvancedObs
from src.rewards.combined_reward import CombinedReward


def create_env(config):
    """Crée 1 environnement RLGym"""
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
        termination_cond=GoalCondition(),  # CORRIGÉ: séparé
        truncation_cond=TimeoutCondition(timeout_seconds=env_config['timeout_seconds']),  # CORRIGÉ
        transition_engine=RocketSimEngine(),
        renderer=None
    )

    return env


def worker(proc_idx, config, conn):
    """
    Worker process qui tourne en parallèle.
    Communique via Pipe avec le process principal.
    """
    env = create_env(config)

    # Reset initial
    result = env.reset()
    obs_dict = result[0] if isinstance(result, tuple) else result

    # Agent ID (typiquement 0 pour 1v1)
    agent_id = list(obs_dict.keys())[0] if isinstance(obs_dict, dict) else 0
    obs = list(obs_dict.values())[0] if isinstance(obs_dict, dict) else obs_dict

    # Episode tracking
    ep_reward = 0.0
    ep_length = 0

    # Boucle de communication
    while True:
        try:
            cmd, data = conn.recv()

            if cmd == "step":
                action = data  # int pour LookupTable
                action_dict = {agent_id: np.array([int(action)])}

                # Step
                step_result = env.step(action_dict)

                if len(step_result) == 5:
                    obs_dict, rew_dict, term_dict, trunc_dict, info = step_result
                else:
                    obs_dict, rew_dict, term_dict, trunc_dict = step_result
                    info = {}

                # Extraire valeurs
                obs = list(obs_dict.values())[0]
                reward = float(list(rew_dict.values())[0])
                terminated = bool(list(term_dict.values())[0])
                truncated = bool(list(trunc_dict.values())[0])
                done = terminated or truncated

                ep_reward += reward
                ep_length += 1

                # Propager reward_parts
                if hasattr(env.reward_fn, '_last_reward_parts'):
                    info['reward_parts'] = env.reward_fn._last_reward_parts

                # Si done, ajouter episode info et reset
                if done:
                    info['episode'] = {'r': ep_reward, 'l': ep_length}
                    info['success'] = 1.0 if ep_reward > 50 else 0.0

                    # Reset
                    result = env.reset()
                    obs_dict = result[0] if isinstance(result, tuple) else result
                    obs = list(obs_dict.values())[0] if isinstance(obs_dict, dict) else obs_dict

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
            print(f"Worker {proc_idx} error: {e}")
            break

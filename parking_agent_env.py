from abc import abstractmethod
from typing import Optional

from gym import Env
import numpy as np

import pygame

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env import utils
from gym.utils import seeding


from stable_baselines3 import HerReplayBuffer, SAC
import envs.gradients



class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingAgentEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None, render_mode: Optional[str] = None, seed = 0, feedback = 0, model = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

        self.action = [0, 0]
        self.obs = [0, 0]
        self.seed_num = 0
        self.type = feedback        
        self.model = model
        # self._np_random = seeding.np_random([0]

        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -500,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 0,
            "add_walls": True
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        
        info = super(ParkingAgentEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        self.action = action
        self.obs = obs
    

        return info

    def render(self, mode: str = 'rgb_array'):
        # CUSTOM CODE
        # This is where the main edits for this work are applied 

        if self.viewer:
            act, _ = self.model.predict(self.obs, deterministic=True)

            # Map agent and user actions (note, user input control is NOT exactly the same as agent control)
            # TODO: likely a better way to implement manual_control than existing one from highway-env. For example continous manipulation of turns instead of discrete
            # See handle_continuous_action_event in envs/common/graphics.py for more info
            agent_action = np.array([utils.lmap(act[0], [-1, 1], [-5, 5]),
                                     utils.lmap(act[1], [-1, 1], [-np.deg2rad(45), np.deg2rad(45)])  ])

            user_action = np.array([self.controlled_vehicles[0].action['acceleration'],
                                    self.controlled_vehicles[0].action['steering']])
          
            if self.type == 1: # Verification

                pygame.draw.arc(self.viewer.sim_surface, (195, 195, 0), (450, 450, 100, 100), np.pi + np.pi/2 - self.controlled_vehicles[0].heading, np.pi + np.pi*3/2 - self.controlled_vehicles[0].heading, width=8)
                pygame.draw.arc(self.viewer.sim_surface, (0, 195, 0), (450, 450, 100, 100),
                                max(3*np.pi/2- self.controlled_vehicles[0].heading,
                                3*np.pi/2 - self.controlled_vehicles[0].heading + np.pi*((-act[1]+1)/2) - np.pi/6), 
                                min(np.pi*5/2 - self.controlled_vehicles[0].heading, 
                                3*np.pi/2 - self.controlled_vehicles[0].heading + np.pi*((-act[1]+1)/2) + np.pi/6), width=8)
                pygame.draw.arc(self.viewer.sim_surface, (195, 195, 195), (450, 450, 100, 100), 
                                np.pi - self.controlled_vehicles[0].heading + np.pi*99/100, 
                                np.pi - self.controlled_vehicles[0].heading + np.pi*101/100,  width=20)

            if self.type == 2: # Correct Response

                if agent_action[1] < -0.1 and user_action[1] > -0.1: # Left
                    pygame.draw.polygon(self.viewer.sim_surface, (195, 195, 0, 255), ( (950,50), (950, 950), (850, 500) )  )
                elif agent_action[1] > -0.1 and user_action[1] < -0.1: # Right
                    pygame.draw.polygon(self.viewer.sim_surface, (195, 195, 0, 255), ( (50,50), (50, 950), (150, 500) )  )
                if agent_action[0] > user_action[0] and user_action[0] < 0.5: # Up
                    pygame.draw.polygon(self.viewer.sim_surface, (195, 195, 0, 255), ( (50,950), (950, 950), (500, 850) )  )
                elif agent_action[0] < user_action[0]: # Down
                    pygame.draw.polygon(self.viewer.sim_surface, (195, 195, 0, 255), ( (50,50), (950, 50), (500, 150) )  )
                    pass
            if not self.viewer.offscreen:
                self.viewer.screen.blit(self.viewer.sim_surface, (0, 0))                

                if self.type == 3: # Error Flagging

                    scol = (195, 195, 0, 255)
                    ecol = (100, 100, 100, 0)                
                    diff = abs(agent_action[1] - user_action[1])
                    if ((abs(agent_action[1] - user_action[1]) > 0.2) and abs(self.controlled_vehicles[0].speed) > 0.1):
                        if abs(agent_action[1] - user_action[1]) > 0.8:
                            scol = (195, 0, 0, 255)
                        self.viewer.screen.blit(envs.gradients.horizontal((int(100 * diff/0.4), 1000), scol, ecol), (0,0))
                        self.viewer.screen.blit(envs.gradients.horizontal((int(100 * diff/0.4), 1000), ecol, scol), (1000 - int(100 * diff/0.4),0))
                        self.viewer.screen.blit(envs.gradients.vertical((1000, int( 100 * diff/0.4)), scol, ecol), (0, 0))
                        self.viewer.screen.blit(envs.gradients.vertical((1000, int( 100 * diff/0.4)), ecol, scol), (0, 1000 - int(100 * diff/0.4)))
            
            # /CUSTOM CODE

                if self.config["real_time_rendering"]:
                    self.viewer.clock.tick(self.config["simulation_frequency"])
                pygame.display.flip()

        super(ParkingAgentEnv, self).render(mode)

        
  
    def _reset(self, seed=2):

        self._create_road()
        self._create_vehicles()
        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        self.my_font = pygame.font.SysFont('Comic Sans MS', 30)

    def _create_road(self, spots: int = 14) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        
        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, [i*20, 0], 2*np.pi*self.np_random.uniform(), 0)
            vehicle.color = (0, 50, 200) # default green
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        # Goal
        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)


        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            lane = ("a", "b", i) if self.np_random.uniform() >= 0.5 else ("b", "c", i)
            v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
            self.road.vehicles.append(v)
        for v in self.road.vehicles:
            if v is not self.vehicle and (
                    np.linalg.norm(v.position - self.goal.position) < 20 or
                    np.linalg.norm(v.position - self.vehicle.position) < 20):
                self.road.vehicles.remove(v)

        # Walls
        for y in [-21, 21]:
            obstacle = Obstacle(self.road, [0, y])
            obstacle.LENGTH, obstacle.WIDTH = (70, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)
        for x in [-35, 35]:
            obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
            obstacle.LENGTH, obstacle.WIDTH = (42, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)


    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        timeout = self.time >= self.config["duration"]
        return bool(crashed or success or timeout)

    def _is_truncated(self) -> bool:
        return False

    def gradientRect(self, window, left_colour, right_colour, target_rect ):
        """ Draw a horizontal-gradient filled rectangle covering <target_rect> """
        colour_rect = pygame.Surface( ( 2, 2 ) )                                   # tiny! 2x2 bitmap
        pygame.draw.line( colour_rect, left_colour,  ( 0,0 ), ( 0,1 ) )            # left colour line
        pygame.draw.line( colour_rect, right_colour, ( 1,0 ), ( 1,1 ) )            # right colour line
        colour_rect = pygame.transform.smoothscale( colour_rect, ( target_rect.width, target_rect.height ) )  # stretch!
        window.blit( colour_rect, target_rect )       


class ParkingEnvActionRepeat(ParkingAgentEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParkingEnvParkedVehicles(ParkingAgentEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10})

import gym
from gym import spaces
import numpy as np


class WaterTankEnv(gym.Env):
    def __init__(self, tank_capacity=100, initial_water_level=50, target_water_level=70):
        self.tank_capacity = tank_capacity
        self.initial_water_level = initial_water_level
        self.target_water_level = target_water_level

        # Define the action and observation space
        self.action_space = spaces.Discrete(2)  # Two actions: 0 (do nothing), 1 (add water)
        self.observation_space = spaces.Box(low=0, high=tank_capacity, shape=(1,),
                                            dtype=np.float32)  # Single continuous state space

        # Initialize the water level
        self.water_level = initial_water_level

    def reset(self):
        # Reset the environment
        self.water_level = np.float32(self.initial_water_level)
        return np.array([self.water_level])

    def step(self, action):
        # Take a step in the environment
        self.water_level += -1
        if action == 1:  # Add water
            self.water_level += 5

        # Clip the water level within the bounds of the tank capacity
        self.water_level = np.clip(self.water_level, 0, self.tank_capacity)

        # Calculate the reward
        reward = -np.abs(
            self.water_level - self.target_water_level)  # Negative reward proportional to distance from target level

        # Check if episode is done
        done = False
        if np.abs(self.water_level - self.target_water_level) < 0.1:
            done = True
        self.water_level = np.float32(self.water_level)

        return np.array([self.water_level]), reward, done, {}

    def render(self, mode='human'):
        # Render the environment
        print(f"Water Level: {self.water_level}")

    def close(self):
        # Clean up resources
        pass


# Register the custom environment
gym.register(id='WaterTankEnv-v0', entry_point=WaterTankEnv)

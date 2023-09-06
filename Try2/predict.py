from TD3 import TD3
from TD3 import Actor
from Environment import TankLevel
import torch
import numpy as np
from PID_controller import PID

pid = PID()

# Initialize the actor model

SP = 0.7
initial_height = 0.3

env = TankLevel(SP=SP, initial_height=initial_height)

max_action = float(env.action_space.high[0])

# ACTOR MODEL PREDICTED ACTION
actor_model = Actor(state_dim=1, action_dim=3, max_action=max_action)
actor_model.load_state_dict(torch.load("E:\\Bali\\RL research\\Try 2\\models\\TD3_TankLevel-v0_0_actor"))

RL_state = env.get_rl_height(height=initial_height, SP=SP)
state = np.array(RL_state)
state = torch.FloatTensor(state.reshape(1, -1))

RL_action = actor_model(state).cpu().data.numpy().flatten()
rl_water_flow_RL_actionconverted = pid.compute(RL_State=state, action=RL_action)
input_flow = env.get_input_flow(rl_water_flow_RL_actionconverted)

## ACTION TAKEN, RESULTED STATE (WATER LEVEL), REWARD

rl_height, reward, done, _ = env.step(Rl_action_3d=RL_action, state=state)
height = env.get_steady_state_height(input_flow)
height_after_1_sec, reward1, done, _  = env.test_step(Rl_action_3d=RL_action,
                                   initial_Rl_state=state)
height_after_1_sec = height_after_1_sec + SP
print("Initial_Height:", initial_height, "Set Point:", SP,
      "Height after action(stady state height reached eventually):", height
      , "height after 10 sec:", height_after_1_sec)

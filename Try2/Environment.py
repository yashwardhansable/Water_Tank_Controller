import numpy as np
import gym
from gym import spaces
import argparse
import utils
import TD3
import torch
from gym.envs.registration import register
import os
import math
import matplotlib.pyplot as plt

from PID_controller import PID


def average_bins(data, bin_length):
    num_bins = math.ceil(len(data) / bin_length)  # Calculate the number of bins
    remaining = bin_length * num_bins - len(data)  # Number of remaining values after division

    averages = []
    start = 0
    end = bin_length

    for i in range(num_bins):
        if remaining > 0:
            end += 1
            remaining -= 1

        bin_values = data[start:end]
        average = sum(bin_values) / len(bin_values)
        averages.append(average)

        start = end
        end += bin_length

    return averages


def plot_graph(data):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds to filter outliers
    lower_bound = -1000
    upper_bound = 500

    # Filter out the outliers
    filtered_data = [value for value in data if lower_bound <= value <= upper_bound]

    # Plot the filtered data
    x = range(len(filtered_data))
    y = filtered_data
    print(filtered_data)

    plt.plot(x, y)
    plt.xlabel('Episodes (in thousands)')
    plt.ylabel('Average reward per thousand Episode')
    plt.title('Graph')
    plt.show()


def eval_policy(policy, eval_episodes=1000):
    eval_env = TankLevel(SP, initial_height)
    # eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(SP, initial_height), False
        while not done:
            action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise,
                                                                               size=action_dim)).clip(-max_action,
                                                                                                      max_action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
        print(state)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
    print("---------------------------------------")
    return avg_reward


class TankLevel():
    def __init__(self, SP, initial_height, max_input_flow=0.014, tank_height=1.2):
        self.SP = np.float64(SP)
        self.initial_height = np.float64(initial_height)
        self.height = self.initial_height  # initial height of the system
        self.input_water_flow = np.float64(0.0)
        self.rl_height = self.get_rl_height(height=self.height, SP=self.SP)  # initial rl height (state0) = -0.4
        self.rl_action = np.float64(0.0)  # rl action_initialization
        self.max_input_flow = np.float64(max_input_flow)

        self.done = False  # done flag

        self.reward = np.float64(0)  # initializing reward

        self.observation_space = spaces.Box(low=-tank_height, high=tank_height, shape=(1,))
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]),
                                       dtype=np.float64)
        self.Rl_action_3d = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.pid = PID()

    def reset(self, SP, initial_height):
        """
        Resets the RL space
        :param SP: State Point
        :param initial_height: initial height
        :return: RL State (RL height)
        """
        self.SP = np.float64(SP)
        self.initial_height = np.float64(initial_height)
        self.done = False

        self.pid.setpoint = np.float64(0.00)
        self.pid.error = np.float64(0.00)
        self.pid.integral_error = np.float64(0.00)
        self.pid.error_last = np.float64(0.00)
        self.pid.derivative_error = np.float64(0.00)
        self.pid.output = np.float64(0.00)

        self.rl_height = self.initial_height - self.SP
        # Reset the environment to its initial state
        # Return the initial observation
        return self.rl_height

    def step(self, state, Rl_action_3d):
        """
        :param action: RL action_3d
        state: initial RL state of the system, used for PID controller
        :return: Next RL State (RL Height) ,reward, done flag
        """
        self.Rl_action_3d = np.float64(Rl_action_3d)

        rl_water_flow_RL_actionconverted = self.pid.compute(RL_State=np.float64(state), action=self.Rl_action_3d)
        # print(f"rl_water_flow_RL_actionconverted: {rl_water_flow_RL_actionconverted}")

        self.input_water_flow = self.get_input_flow(
            rl_water_flow_RL_actionconverted)  # returns real flow when action is taken
        # print(f"self.input_water_flow: {self.input_water_flow}")

        self.height = self.get_steady_state_height(
            self.input_water_flow)  # steady state height if water is left by itself
        # print(f"self.height: {self.height}")

        self.reward = self.get_reward(self.height)  # reward obtained at height steady state
        # print(f"self.reward: {self.reward}")

        self.rl_height = self.get_rl_height(self.height,
                                            self.SP)  # difference between steady state height and actual height ??
        # print(f"rl_height:{self.rl_height}")

        if 0.1 > self.rl_height > -0.1:  # Check if self.rl_height is close to 0
            self.done = True
            return self.rl_height, self.reward, self.done, {}

        # Take an action and update the environment state
        # Return the new observation, reward, done flag, and info dictionary
        return self.rl_height, self.reward, self.done, {}

    def render(self):
        # Render the current state of the environment
        pass

    def get_reward(self, rl_height):
        """

        :param rl_height: RL_height
        :return: Reward
        """
        S = np.round(self.rl_height, 1)
        if -0.1 < S < 0.1:
            reward = 100
        else:
            reward = -np.square(S) * 100
        return reward

    def get_input_flow(self, action):
        """

        :param action: RL_action
        :return: real water flow
        """
        self.rl_action = action  # action taken
        S = max(0, self.input_water_flow + self.rl_action)
        S = min(self.max_input_flow, S)  # input water flow
        return S

    def get_steady_state_height(self, input_flow):
        """

        :param input_flow: real_input flow of the system
        :return: height which will be achieved if water keeps flowing at current rate (steady state)
        """
        S = np.divide(np.square(input_flow), np.square(0.0133))
        return S

    def get_rl_height(self, height, SP):
        """

        :param height: Height of the system
        :param SP: State Point
        :return: RL_Height
        """
        return np.round(height - SP, 2)

    def close(self):
        # Clean up any resources used by the environment
        pass

    def test_step(self, Rl_action_3d, initial_Rl_state):
        '''

        Rl_action_3d: action taken by the RL agent
        initial_Rl_state: initial RLstate of the system
        :return: RL_state, reward, done, {}
        '''
        self.Rl_action_3d = Rl_action_3d
        self.rl_height = initial_Rl_state

        rl_water_flow_RL_actionconverted = self.pid.compute(RL_State=self.rl_height, action=self.Rl_action_3d)
        self.input_water_flow = self.get_input_flow(rl_water_flow_RL_actionconverted)
        Ts = 0.1  # timesstep on which update recursively

        self.height = np.round(self.rl_height + self.SP, 6)
        self.height = np.clip(self.height, 0.0, 1.2)
        for i in range(100):
            self.height = self.height + (Ts / 0.79) * (self.input_water_flow - 0.0133 * np.sqrt(max(self.height, 0)))
            self.height = np.round(self.height, 6)
        self.height = np.clip(self.height, 0.0, 1.2)

        self.reward = self.get_reward(self.height)

        self.rl_height = self.get_rl_height(height=self.height, SP=self.SP)

        if 0.1 > self.rl_height > -0.1:  # Check if self.rl_height is close to 0
            self.done = True
            return self.rl_height, self.reward, self.done, {}

        # Take an action and update the environment state
        # Return the new observation, reward, done flag, and info dictionary
        return self.rl_height, self.reward, self.done, {}


Training = True

if Training:

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="TankLevel-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    SP = np.float64(0.6)
    initial_height = np.float64(1.0)

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = TankLevel(SP, initial_height)

    # Set seeds
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("max_action:", max_action)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(SP, initial_height), False
    Rl_action_3d = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    episode_reward = np.float64(0)
    episode_timesteps = 0
    episode_num = 0

    episode_rewards = []
    reward_per_hundred_ep = 0.0

    for episode in range(100):  # Training for 1000 episodes

        if episode % 1000 == 0:
            # print("Average_Reward_per_thousand_episodes:", reward_per_hundred_ep/100)

            reward_per_hundred_ep = 0.00

        if episode > 0 and episode % 10 == 0:
            sp_range = list(np.round(np.linspace(0, 1.2, 21), 1))
            SP = np.float64(np.random.choice(sp_range, 1))
            initial_height = np.float64(np.random.choice(sp_range, 1))
            state = env.reset(SP=SP, initial_height=initial_height)

        else:
            state = env.reset(SP=SP, initial_height=initial_height)

        initial_state = state
        Set_Point = SP

        episode_reward = 0

        # print("state",state)
        # print(f"Episode{episode+1} : Initial_state = {initial_state}")

        for t in range(1000):
            Rl_action_3d = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise,
                                                                                     size=action_dim)).clip(-max_action,
                                                                                                            max_action)

            next_state, reward, done, _ = env.step(state=state, Rl_action_3d=Rl_action_3d)
            #print(env.SP, env.height)
            #print(env.test_step(initial_Rl_state=state, Rl_action_3d=Rl_action_3d))

            replay_buffer.add(state, Rl_action_3d, next_state, reward, done)
            #print("state:",state)
            #print("action:", Rl_action_3d)
            #print("reward:", reward)
            #print("next_state:",next_state)
            #print("height:", env.height)
            # print("done:", done)
            #print(f"Integral_error = {env.pid.integral_error}, derivative_error = {env.pid.derivative_error}, Error = {env.pid.error}, error_last ={env.pid.error_last}")
            # print(f"State: {state}, action: {Rl_action_3d},next_state:{next_state}, reward: {reward}")
            # print(f" P: {env.pid.kp} I:{env.pid.ki} D:{env.pid.kd} ")

            episode_reward += reward
            state = next_state
            if t >= 200:
                policy.train(replay_buffer, args.batch_size)
                # print("training...")

            if done:
                episode_rewards.append(episode_reward)
                # print("final_state", state)
                # print(f"Episode {episode + 1}: Reward = {episode_reward}")
                break

        # print("hit")
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Initial_State = {initial_state}, Final_State = {state},Set Point = {Set_Point}")

        reward_per_hundred_ep += episode_reward
        # print("hi")
    print("P:", env.pid.get_kpe(), "I:", env.pid.get_kie(), "D:", env.pid.get_kde())
    policy.save(f"./models/{file_name}")

    data = episode_rewards
    bin_length = 100

    averages_rewards = average_bins(data, bin_length)

    data = averages_rewards
    plot_graph(data)

    env.close()



water_tank = TankLevel(1.2, 0.3)
print(water_tank.test_step([0.22, 0.33, 0.44], 0.15))
from TD3 import TD3
from Environment import WaterTankEnvironment
from ReplayBuffer import ReplayBuffer
import numpy as np
import torch
import math
import matplotlib.pyplot as plt


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
    lower_bound = -100000
    upper_bound = 10000

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


SP = 5.0
current_level = 3.0
max_action = 1.0
expl_noise = 0.1
action_dim = 3
state_dim = 1
batch_size = 256

env = WaterTankEnvironment(target_level=SP, current_level=3.0)
buffer_size = 512
replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

# Fill the replay buffer with random experiences
state = env.reset(target_level=SP, current_level=current_level)

TD3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)

file_name = "Try3"
episode_rewards = []

for episode in range(1000):  # Training for 1000 episodes

    if episode > 0 and episode % 10 == 0:
        sp_range = list(np.round(np.linspace(0, 10, 21), 1))
        SP = np.float64(np.random.choice(sp_range, 1))
        initial_height = np.float64(np.random.choice(sp_range, 1))
        state = env.reset(target_level=SP, current_level=current_level)

    else:
        state = env.reset(target_level=SP, current_level=current_level)

    initial_state = state
    Set_Point = SP

    episode_reward = 0

    # print("state",state)
    # print(f"Episode{episode+1} : Initial_state = {initial_state}")

    for t in range(1000):
        action = (TD3.select_action(state=np.array(state)) + np.random.normal(0, max_action * expl_noise,
                                                                              size=action_dim)).clip(-max_action,
                                                                                                     max_action)

        next_state, reward, done = env.step(action=action)

        replay_buffer.add(state, action, next_state, reward, done)

        episode_reward += reward
        state = next_state
        if t >= 200:
            TD3.train(replay_buffer, batch_size)
            #print("training...")

        if done:
            episode_rewards.append(episode_reward)
            # print("final_state", state)
            # print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break

    # print("hit")
    print(
        f"Episode {episode + 1}: Reward = {episode_reward}, Initial_State = {initial_state}, Final_State = {state},Set Point = {Set_Point}")

    # print("hi")
print(env.p, env.i, env.d)
# error = env.step([env.p, env.i, env.d])
# target_level = 2.0
# print(target_level - error)
TD3.save(f"./models/{file_name}")

data = episode_rewards
bin_length = 100

averages_rewards = average_bins(data, bin_length)

data = averages_rewards
plot_graph(data)

# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


Water_tankEx = WaterTankEnvironment(6.00, 2.00)
error = env.step([env.p, env.i, env.d])

print(f"P:{env.p}, I: {env.i}, D: {env.d}")
print(f"Arbitrary initial_height: 2.00, Target_Height: 6.00, height after 1 timestep: {6.00 - error[0]}")




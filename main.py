import gym
from Environment import WaterTankEnv
import tensorflow as tf
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add_transition(self, state, action, reward, next_state):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[idx] for idx in indices]
        return zip(*batch)


class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.q_value = tf.keras.layers.Dense(1)

    def call(self, states, actions):
        # print(action)
        # action = tf.reshape(action, (-1, 1))
        # print(state)
        # print(action)
        states = tf.convert_to_tensor(states, dtype= tf.float32)
        actions = tf.cast(actions,dtype=tf.float32)
        if actions.shape != (64, 1):
            actions = tf.reshape(actions, (-1, 1))

        x = tf.concat([states, actions], axis=-1)
        # print(x)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.q_value(x)
        return q_value


class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(2, activation='relu', trainable=True)
        self.fc2 = tf.keras.layers.Dense(2, activation='relu', trainable=True)
        self.action_probs = tf.keras.layers.Dense(action_dim, activation='softmax', trainable=True)

    def call(self, state):
        state = tf.convert_to_tensor(state, dtype= tf.float32)
        x = self.fc1(state)
        x = self.fc2(x)
        action_probs = self.action_probs(x)
        action_probs = tf.round(action_probs * 1000) / 1000
        action_probs = action_probs / tf.reduce_sum(action_probs, axis=1, keepdims=True)
        return action_probs


class TD3Agent:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, discount, tau,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_critic1 = CriticNetwork(state_dim, action_dim)
        self.target_critic2 = CriticNetwork(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype = tf.float32)
        action_probs = self.actor(state)
        action_probs = action_probs / tf.reduce_sum(action_probs, axis=1, keepdims=True)
        pr = action_probs[0].numpy().tolist()
        pr = pr / np.sum(pr, keepdims=True)
        # print(pr)

        action = np.random.choice(self.action_dim, p=pr)
        return action

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # print("hi1")

        states, actions, rewards, next_states = self.replay_buffer.sample_batch(self.batch_size)
        print("states:",states)
        print("actions:", actions)
        print("rewards:", rewards)
        print("next_states:", next_states)

        # print(self.replay_buffer.sample_batch(self.batch_size))

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        # print(next_states)
        states = tf.convert_to_tensor(states, dtype= tf.float32)
        actions = tf.convert_to_tensor(actions, dtype= tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)


        next_actions = self.target_actor.call(next_states)
        print("next_actions", next_actions)

        # print("hi2")
        # next_actions += np.clip(np.random.normal(0, self.policy_noise, size=next_actions.shape), -self.noise_clip, self.noise_clip)
        # print(next_actions)
        next_actions = tf.random.categorical(tf.math.log(next_actions), num_samples=1)
        # print(next_actions)
        target_q_values1 = self.target_critic1.call(next_states, next_actions)
        target_q_values2 = self.target_critic2.call(next_states, next_actions)
        # print("hi3")
        target_q_values = np.minimum(target_q_values1, target_q_values2)
        target_q_values = rewards + self.discount * target_q_values
        # print("hi4")




        with tf.GradientTape() as tape:
            q_values1 = self.critic1.call(states, actions)
            #target_q_values = tf.convert_to_tensor(target_q_values)
            #q_values1 = tf.convert_to_tensor(q_values1)
            critic1_loss = tf.reduce_mean(tf.square(target_q_values - q_values1))

        with tf.GradientTape() as tape2:
            q_values2 = self.critic2.call(states, actions)
            #target_q_values = tf.convert_to_tensor(target_q_values)
            #q_values2 = tf.convert_to_tensor(q_values2)
            critic2_loss = tf.reduce_mean(tf.square(target_q_values - q_values2))

        critic1_gradients = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_gradients = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

        if self.policy_delay % self.policy_delay == 0:
            with tf.GradientTape() as tape3:
                states = tf.convert_to_tensor(states)
                action_probs = self.actor(states)
                print("states:", states)
                print("action_prob", action_probs)
                log_probs = tf.math.log(action_probs)
                print("actions:", actions)
                one_hot = tf.one_hot(tf.cast(actions,dtype= tf.int32), depth= self.action_dim)
                selected_log_probs = tf.reduce_sum(log_probs * one_hot, axis=1)
                print("q_values1:", q_values1)
                print("selected_log_probs:",selected_log_probs)
                actor_loss = -tf.reduce_mean(selected_log_probs * q_values1)
                #print("actor.trainable_variables:", self.actor.trainable_variables)

            actor_gradients = tape3.gradient(actor_loss, self.actor.trainable_variables)
            trainable_variables = [var for var in self.actor.trainable_variables if var.trainable]
            print("trainable variable",trainable_variables)
            print("actor_gradients:", actor_gradients)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.update_target_networks()

    def update_target_networks(self):
        target_actor_weights = []
        actor_weights = self.actor.get_weights()
        for target_weight, weight in zip(self.target_actor.get_weights(), actor_weights):
            target_actor_weights.append(self.tau * weight + (1 - self.tau) * target_weight)
        self.target_actor.set_weights(target_actor_weights)

        target_critic1_weights = []
        critic1_weights = self.critic1.get_weights()
        for target_weight, weight in zip(self.target_critic1.get_weights(), critic1_weights):
            target_critic1_weights.append(self.tau * weight + (1 - self.tau) * target_weight)
        self.target_critic1.set_weights(target_critic1_weights)

        target_critic2_weights = []
        critic2_weights = self.critic2.get_weights()
        for target_weight, weight in zip(self.target_critic2.get_weights(), critic2_weights):
            target_critic2_weights.append(self.tau * weight + (1 - self.tau) * target_weight)
        self.target_critic2.set_weights(target_critic2_weights)


# Training loop


# Create an instance of the water tank environment
gym.register(id='WaterTankEnv-v0', entry_point=WaterTankEnv)
env = gym.make('WaterTankEnv-v0')
observation = env.reset()

# Set the required parameters
state_dim = env.observation_space.shape[0]
action_dim = 2
buffer_size = 100
batch_size = 4
gamma = 0.99
tau = 0.001
actor_lr = 0.001
critic_lr = 0.001
num_episodes = 100
max_steps = 1000
hidden_dim = 256

# if __name__ == '__main__':
#     agent = TD3Agent(state_dim, action_dim, buffer_size, batch_size, gamma, tau, 0.2, 0.5, 2)
#     state = env.reset()
#     for _ in range(265):
#         action = agent.select_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.replay_buffer.add_transition(state, action, reward, next_state)
#         agent.train()
#         state = next_state
#
#     #print(len(agent.replay_buffer.buffer))


if __name__ == '__main__':
    agent = TD3Agent(state_dim, action_dim, buffer_size, batch_size, gamma, tau, 0.2, 0.5, 2)
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add_transition(state, action, reward, next_state)
            agent.train()

            total_reward += reward
            state = next_state

            if done:
                print("Episode:", episode, "Total Reward:", total_reward)
                break

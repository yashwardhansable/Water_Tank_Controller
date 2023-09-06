import torch

class WaterTankEnvironment:
    def __init__(self, target_level, current_level):
        self.target_level = target_level
        self.current_level = current_level
        self.max_level = 10.0
        self.min_level = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.p = 0.0
        self.i = 0.0
        self.d = 0.0

    def reset(self, target_level, current_level):
        self.current_level = current_level
        self.integral = 0.0
        self.prev_error = 0.0
        self.target_level = target_level
        done = False
        return self.target_level - self.current_level


    def step(self,action):
        p, i, d = action
        self.p = p
        self.i = i
        self.d = d
        error = self.target_level - self.current_level

        # Calculate the proportional component
        proportional = p * error

        # Calculate the integral component
        self.integral += i * error
        integral = self.integral

        # Calculate the derivative component
        derivative = d * (error - self.prev_error)
        #print(proportional, integral,derivative)

        # Update the control signal
        control_signal = proportional + integral + derivative

        # Update the water level
        self.current_level += control_signal

        # Clip the water level within the valid range
        self.current_level = max(self.min_level, min(self.current_level, self.max_level))

        # Calculate the reward based on the control performance
        reward = -abs(error)*1000  # Negative absolute error as penalty

        # Set the previous error for the next step
        self.prev_error = error

        # Determine if the episode is done (optional)
        done = False
        if(abs(error) < 0.0001):
            reward += 10000
            done = True

        return self.prev_error, reward, done


# # Create the water tank environment
# target_level = 5.0
# max_capacity = 10.0
# num_steps = 10000
# current_level = 2000.0
# env = WaterTankEnvironment(target_level, current_level= current_level)
#
# # Reset the environment
# env.reset(target_level=target_level, current_level=current_level)
#
# # Perform steps in the environment
#  # Example control signal, you can change this
# for _ in range(num_steps):
#     state, reward, done = env.step(torch.tensor([0.1, 0.2, 0.3]))
#     print(state, reward, done)
#     # Update the control signal based on the learned policy
#     # Perform other necessary operations
#
#     if done:
#         break

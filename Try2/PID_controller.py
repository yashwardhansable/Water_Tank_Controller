import numpy as np


class PID(object):
    def __init__(self):

        self.setpoint = np.float64(0.00)
        self.error = np.float64(0.00)
        self.integral_error = np.float64(0.00)
        self.error_last = np.float64(0.00)
        self.derivative_error = np.float64(0.00)
        self.output = np.float64(0.00)

    def compute(self, RL_State, action):
        """
        :param (P, I, D): actions obtained through TD3
        :return: RL_state flow rate
        """
        self.kp = np.float64(action[0])
        self.ki = np.float64(action[1])
        self.kd = np.float64(action[2])
        self.error = np.float64(RL_State) - self.setpoint
        self.integral_error += self.error * 10
        self.derivative_error = (self.error - self.error_last)/10
        self.error_last = self.error
        self.output = self.kp * self.error + self.ki * self.integral_error + self.kd * self.derivative_error

        if self.output < -0.014:
            self.output = -0.014
        if self.output > 0.014:
            self.output = 0.014

        return self.output

    def get_kpe(self):
        return self.kp #* self.error

    def get_kde(self):
        return self.kd #* self.derivative_error

    def get_kie(self):
        return self.ki #* self.integral_error

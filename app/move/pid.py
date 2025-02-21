import time


class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=float('inf')):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()

    def reset(self):
        """重置积分和微分项"""
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()

    def compute(self, error):
        """
        pid计算移动幅度
        :param error: 误差
        :return: 移动幅度
        """
        current_time = time.time()
        delta_time = current_time - self.previous_time
        if delta_time <= 0:
            delta_time = 0.01

        # 计算积分项并限制
        self.integral += error * delta_time
        # 计算微分项
        derivative = (error - self.previous_error) / delta_time if delta_time > 0 else 0

        # PID输出
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        # 限制输出范围
        output = max(-self.max_output, min(output, self.max_output))

        self.previous_error = error
        self.previous_time = current_time

        return output
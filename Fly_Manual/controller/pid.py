class PID:
    """Standard PID with derivative-on-measurement and anti-windup clamp."""
    def __init__(self, kp, ki, kd, out_min, out_max, i_min=None, i_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_min = i_min if i_min is not None else out_min
        self.i_max = i_max if i_max is not None else out_max
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def update(self, error, rate, dt):
        """rate = derivative of the *measurement* (not the error), pass in already
        signed so that d(error)/dt = -rate for a constant setpoint."""
        self._integral += error * dt
        self._integral = max(self.i_min, min(self.i_max, self._integral))
        out = self.kp * error - self.kd * rate + self.ki * self._integral
        return max(self.out_min, min(self.out_max, out))

import math


def exp_decay(t, t_max, initial_value):
    return initial_value * math.exp(-t / t_max)


def linear_decay(t, initial_value):
    return initial_value / (t + 1)

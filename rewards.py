from tolerances import tolerance
import math
import numpy as np

Height = 0.6

# mode = 0 NOT SPARSE
def compute_simplest_reward(state):
    x, cos_theta, sin_theta, x_vel, theta_vel, x_0, y_0, mass = state

    return (cos_theta +1)/2 * math.cos(x-x_0)

# mode = 1 SPARSE (not recommended)
def compute_exponential_reward(state):
    x, cos_theta, sin_theta, x_vel, theta_vel, x_0, y_0 , mass= state
    x_cur = x + Height * sin_theta
    y_cur = Height * cos_theta
    Dist = (x_cur - x_0)**2 + (y_cur - y_0)**2
    cost = 1 - math.exp(-Dist / (2 * 0.25**2))

    return -cost

# mode = 2 SPARSE (not recommended)
def compute_simplest_tolerance_reward(state):

    x, cos_theta, sin_theta, x_vel, theta_vel, x_0, y_0, mass = state
    cart_in_bounds = tolerance(x, (x_0-0.15, x_0+0.15))
    angle_in_bounds = tolerance(cos_theta,(0.99, 1))

    return cart_in_bounds * angle_in_bounds


# mode = 3 NOT SPARSE
def compute_tolerance_reward(state, action):

    x, cos_theta, sin_theta, x_vel, theta_vel, x_0, y_0, mass = state
    upright = (cos_theta + 1) / 2
    centered = tolerance(x, bounds=(x_0,x_0), margin=1) 
    centered = (1 + centered) / 2
    small_control = tolerance(action, margin=3.0, value_at_margin=0, sigmoid='quadratic')[0]
    small_control = (4 + small_control) / 5
    small_velocity = tolerance(theta_vel, margin=5.0)
    small_velocity = (1 + small_velocity) / 2
        
    reward = upright * small_control * small_velocity * centered

    return reward

# mode = 4 NOT SPARSE (recommended)
def compute_tolerance_reward_xy(state, action):

    x, cos_theta, sin_theta, x_vel, theta_vel, x_0, y_0, mass = state
    upright = (cos_theta + 1) / 2
    centered = tolerance(x, bounds=(x_0,x_0), margin=3.0) 
    centered = (1 + centered) / 2

    centered_pole = tolerance(x+sin_theta*Height, bounds=(x_0,x_0), margin=0.5) 
    centered_pole = (1 + centered_pole) / 2

    up_pole = tolerance(cos_theta*Height, bounds=(Height,Height), margin=0.5) 
    up_pole = (1 + up_pole) / 2

    small_control = tolerance(action, margin=1.0, value_at_margin=0, sigmoid='quadratic')[0]
    small_control = (4 + small_control) / 5
    small_velocity = tolerance(theta_vel, margin=5.0)
    small_velocity = (1 + small_velocity) / 2
        
    reward = upright * small_control * small_velocity * centered * centered_pole*up_pole

    return reward


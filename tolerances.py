import warnings
import numpy as np

'''Taken as reference from https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/rewards.py '''

def _sigmoids(x, value_at_1, sigmoid):

  if sigmoid in ('cosine', 'linear', 'quadratic'):
    if not 0 <= value_at_1 < 1:
      raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                       'got {}.'.format(value_at_1))
  else:
    if not 0 < value_at_1 < 1:
      raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                       'got {}.'.format(value_at_1))

  if sigmoid == 'gaussian':
    scale = np.sqrt(-2 * np.log(value_at_1))
    return np.exp(-0.5 * (x*scale)**2)

  elif sigmoid == 'hyperbolic':
    scale = np.arccosh(1/value_at_1)
    return 1 / np.cosh(x*scale)

  elif sigmoid == 'long_tail':
    scale = np.sqrt(1/value_at_1 - 1)
    return 1 / ((x*scale)**2 + 1)

  elif sigmoid == 'reciprocal':
    scale = 1/value_at_1 - 1
    return 1 / (abs(x)*scale + 1)

  elif sigmoid == 'cosine':
    scale = np.arccos(2*value_at_1 - 1) / np.pi
    scaled_x = x*scale
    with warnings.catch_warnings():
      warnings.filterwarnings(
          action='ignore', message='invalid value encountered in cos')
      cos_pi_scaled_x = np.cos(np.pi*scaled_x)
    return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x)/2, 0.0)

  elif sigmoid == 'linear':
    scale = 1-value_at_1
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

  elif sigmoid == 'quadratic':
    scale = np.sqrt(1-value_at_1)
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

  elif sigmoid == 'tanh_squared':
    scale = np.arctanh(np.sqrt(1-value_at_1))
    return 1 - np.tanh(x*scale)**2

  else:
    raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='cosine',
              value_at_margin=0.1):

  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = np.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = np.where(in_bounds, 1.0, 0.0)
  else:
    d = np.where(x < lower, lower - x, x - upper) / margin
    value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  return float(value) if np.isscalar(x) else value
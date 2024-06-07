import sys
import time
import numpy as np

from shutil import get_terminal_size

def super_quantile(x, alpha, discretization = .01):
    
    q = np.arange(alpha, .99, discretization)
    # print(q)
    
    sq = 1/(1 - alpha) * (np.quantile(x, q) * discretization).sum()

    return sq

def super_quantile_normal(x, alpha, discretization = .01):
    
    # q = np.arange(alpha, .99, discretization)
    # # print(q)
    
    # sq = 1/(1 - alpha) * (np.quantile(x, q) * discretization).sum()

    return x.mean() + alpha * x.std()

'''
Calculates Gini coefficient (inequality)
'''
def gini(x):

    x = np.array(x)

    total = 0

    for i, xi in enumerate(x[:-1], 1):

        total += np.sum(np.abs(xi - x[i:]))

    return total / (len(x) ** 2 * np.mean(x))

def in_iterable(value):

    return hasattr(value, '__iter__')

def top_n_indices(array, n):

    return sorted(range(len(array)), key=lambda i: array[i])[-n:]

def bottom_n_indices(array, n):

    return sorted(range(len(array)), key=lambda i: array[i])[:n]

def full_factorial(levels):

    n = len(levels)  # number of factors

    nb_lines = np.prod(levels)  # number of trial conditions

    h = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels).astype(int)

    for i in range(n):

        range_repeat /= levels[i]
        range_repeat = range_repeat.astype(int)

        lvl = []

        for j in range(levels[i]):

            lvl += [j] * level_repeat

        rng = lvl*range_repeat

        level_repeat *= levels[i]

        h[:, i] = rng

    return h.astype(int)

def pythagorean(source_x, source_y, target_x, target_y):

    return np.sqrt((target_x - source_x) ** 2 + (target_y - source_y) ** 2)

def haversine(source_longitude, source_latitude, target_longitude, target_latitude, **kwargs):

    radius = kwargs.get('radius', 6372800) # [m]
    
    distance_longitude_radians = np.radians(target_longitude - source_longitude)
    distance_latitude_radians = np.radians(target_latitude - source_latitude)

    source_latitude_radians = np.radians(source_latitude)
    target_latitude_radians = np.radians(target_latitude)

    a_squared = (
        np.sin(distance_latitude_radians / 2) ** 2 +
        np.cos(source_latitude_radians) *
        np.cos(target_latitude_radians) *
        np.sin(distance_longitude_radians / 2) ** 2
        )

    c = 2 * np.arcsin(np.sqrt(a_squared))

    return c * radius

def root_mean_square_error(x, y):

    return np.sqrt(((x - y) ** 2).sum() / len(x))

def cprint(message, disp = True, **kwargs):

    if disp:

        print(message, **kwargs)
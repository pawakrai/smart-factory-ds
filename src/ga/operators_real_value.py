# ฟังก์ชันเพิ่มเติม เช่น crossover, mutation operators

import numpy as np
from copy import deepcopy


def roulette_wheel_selection(prob):
    """Roulette Wheel Selection"""
    c = np.cumsum(prob)
    r = np.random.rand() * c[-1]
    idx = np.argwhere(r <= c)
    return idx[0][0]


def crossover(p1, p2, gamma=0.1):
    """Blend Crossover (BGA) แบบง่าย ๆ"""
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    alpha = np.random.uniform(-gamma, 1 + gamma, p1["position"].shape)
    c1["position"] = alpha * p1["position"] + (1 - alpha) * p2["position"]
    c2["position"] = alpha * p2["position"] + (1 - alpha) * p1["position"]

    return c1, c2


def mutate(ind, mu, sigma):
    """Gaussian Mutation"""
    c = deepcopy(ind)
    mask = np.random.rand(*c["position"].shape) < mu
    c["position"][mask] += sigma * np.random.randn(np.sum(mask))
    return c


def apply_bound(ind, var_min, var_max):
    """Clamp ค่าให้อยู่ใน boundary"""
    ind["position"] = np.maximum(ind["position"], var_min)
    ind["position"] = np.minimum(ind["position"], var_max)

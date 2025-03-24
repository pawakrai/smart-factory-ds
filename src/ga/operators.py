# ga/operators.py

import numpy as np
from copy import deepcopy


def roulette_wheel_selection(prob):
    cumsum = np.cumsum(prob)
    r = np.random.rand() * cumsum[-1]
    idx = np.where(r <= cumsum)[0][0]
    return idx


def crossover(p1, p2, gamma=0.1):
    # Integer crossover: ใช้วิธี uniform crossover สำหรับ integer vector
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)
    for i in range(len(p1["position"])):
        if np.random.rand() < 0.5:
            c1["position"][i], c2["position"][i] = p2["position"][i], p1["position"][i]
    return c1, c2


def mutate(ind, mu, sigma, var_min, var_max):
    # Integer mutation: สำหรับแต่ละ gene ให้สุ่มเปลี่ยนแปลงด้วยความน่าจะเป็น mu
    c = deepcopy(ind)
    n = len(c["position"])
    n_batches = n // 2
    for i in range(n):
        if np.random.rand() < mu:
            # ถ้าเป็นตัวแปร start_slot (indices 0 ถึง n_batches-1)
            if i < n_batches:
                c["position"][i] = np.random.randint(var_min[i], var_max[i] + 1)
            else:
                # สำหรับ furnace assignment: ควรสุ่มเป็น 0 หรือ 1
                c["position"][i] = np.random.randint(0, 2)
    return c


def apply_bound(ind, var_min, var_max):
    ind["position"] = np.maximum(ind["position"], var_min)
    ind["position"] = np.minimum(ind["position"], var_max)

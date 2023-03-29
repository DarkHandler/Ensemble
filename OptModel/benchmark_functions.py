import numpy as np


def F1(x):
    s = np.sum(x ** 2)
    return s

def F6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o



def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F6": ["F6", -100, 100, 30],
        "teamSizeModel":["teamSizeModel", 3, 18, 200],
    }
    return param.get(a, "nothing")
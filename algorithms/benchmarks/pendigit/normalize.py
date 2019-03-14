import numpy as np
from math import sqrt

def normalize_global(data):
    mx = 0
    my = 0
    stx = 0
    sty = 0
    numobs = 0
    for example in data:
        for obs in example:
            if obs[0] == -1:
                continue
            mx += obs[0]
            my += obs[1]
            numobs += 1
    mx = float(mx) / numobs
    my = float(my) / numobs

    for example in data:
        for obs in example:
            if obs[0] == -1:
                continue
            stx += (obs[0] - mx) ** 2
            sty += (obs[1] - my) ** 2

    stx = float(stx) / numobs
    sty = float(sty) / numobs

    data2 = []
    tdata = []
    for example in data:
        trace = []
        for obs in example:
            if obs[0] == -1:
                x = obs[0]
                y = obs[1]
            else:
                x = (obs[0] - mx) / stx
                y = (obs[1] - my) / sty
                tdata.append([x, y])
            trace.append(np.array([x, y]))
        data2.append(np.array(trace))
    return data2, tdata


def normalize_example(data):
    data2 = []
    tdata = []

    for example in data:
        numobs = 0
        mx = 0
        my = 0
        stx = 0
        sty = 0
        for obs in example:
            if obs[0] == -1:
                continue
            mx += obs[0]
            my += obs[1]
            numobs += 1

        mx = int((float(mx) / float(numobs)))
        my = int((float(my) / float(numobs)))
        if mx == 0 or my == 0:
            print("KO1  ")

        for obs in example:
            if obs[0] == -1:
                continue
            stx += (obs[0] - mx) * (obs[0] - mx)
            sty += (obs[1] - my) * (obs[1] - my)

        stx = (float(stx) / float(numobs))
        sty = (float(sty) / float(numobs))


        trace = []
        for obs in example:
            if obs[0] == -1:
                x = obs[0]
                y = obs[1]
            else:
                x = float(obs[0] - mx) / sqrt(stx)
                y = float(obs[1] - my) / sqrt(sty)
                tdata.append([x, y])
            trace.append(np.array([x, y]))

        data2.append(np.array(trace))
    return data2, tdata
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def subplot_heuristic(n):
    ''' for n subplots, determine best grid layout dimensions '''
    def isprime(n):
        for x in range(2, int(np.sqrt(n)) + 1):
            if n % x == 0:
                return False
        return True
    if n > 6 and isprime(n):
        n += 1
    num_lst, den_lst = [n], [1]
    for x in range(2, int(np.sqrt(n)) + 1):
        if n % x == 0:
            den_lst.append(x)
            num_lst.append(n // x)
    ratios = np.array([a / b for a, b in zip(num_lst, den_lst)])
    best_ind = np.argmin(ratios - 1.1618)  # most golden
    if den_lst[best_ind] < num_lst[best_ind]: # always have more rows than cols
        return den_lst[best_ind], num_lst[best_ind]
    else:
        return num_lst[best_ind], den_lst[best_ind]

class MidpointNormalize(colors.Normalize):
    ''' create asymmetric norm '''

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class ImageFollower(object):
    ''' update image in response to changes in clim or cmap on another image '''

    def __init__(self, follower):
        self.follower = follower

    def __call__(self, leader):
        self.follower.set_cmap(leader.get_cmap())
        self.follower.set_clim(leader.get_clim())


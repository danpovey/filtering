#!/usr/bin/env python3
from itertools import count

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



J = 512   # Defines how many discrete points we divide the interval [0..pi] of F(t) into.
kM = 4    # after how far out in time we start penalizing
kN = 8     # how far out in time we compute it.

def get_fourier_matrix(J):
    """ Returns a matrix as a torch.Tensor that maps F_j to f_k.  The row index
        corresponds to k, and the column index to j, so the dimension is
        4k by 2j.
    """
    M = np.empty((kN*J, 2*J), dtype=np.float64)
    for j in range(2*J):
        c = (1.0/J) * (1.0 if j > 0 else 0.5)
        for k in range(kN*J):
            M[k, j] = c * math.cos( (math.pi * j * k) / (J * J))

    return torch.tensor(M, dtype=torch.float64)



def __main__():
    # theta_j will contain theta values from 1 to J-1.  The choice of
    # multiplying by pi / (2*J) means that at j=J we would have theta_j = pi/2;
    # and the cosine and sine of that are the same at sqrt(2), as required for continuity,
    # since theta_J is defined as exactly sqrt(2).
    theta_j = torch.tensor(np.arange(1, J, dtype=np.float64) * (math.pi / (4 * J)),
                           requires_grad = True)


    M = get_fourier_matrix(J)

    # F_j is defined for 0 <= j < 2J.  note: theta_j[J-2::-1] is just theta_j with
    # elements reversed.
    F_j = torch.abs(
        torch.cat((torch.tensor([1.0], dtype=torch.float64),
                   torch.cos(theta_j),
                   torch.tensor([math.sqrt(0.5)], dtype=torch.float64),
                   torch.sin(torch.flip(theta_j, [0])))))
    f_k = torch.mv(M, F_j)
    O = (f_k[2*J : kN*J] ** 2).sum()

    def f_and_O(iter):
        # Returns the tuple (f_k, O) where O is the objective function.
        # This is computed from scratch starting with theta_j.
        F_j = torch.cat((torch.tensor([1.0], dtype=torch.float64),
                       torch.cos(theta_j),
                       torch.tensor([math.sqrt(0.5)], dtype=torch.float64),
                       torch.sin(torch.flip(theta_j, [0])),
                       torch.tensor([0.0], dtype=torch.float64)))

        f_k = torch.mv(M, F_j[:-1])

        f_penalty_scale = torch.cat(
            (torch.exp(torch.linspace(-20, 0, kM*J, dtype=torch.float64)),
             torch.ones( (kN-kM)*J, dtype=torch.float64 )))

        f_penalty = (f_k * f_penalty_scale).abs().sum() * 0.1

        grad_limit = torch.tensor(2.5 / (2*J), dtype=torch.float64)  #  1/(2J) would be avg grad if it fell from 1 to 0 linearly.  The 2.5 is wiggle room;
                                                                    # it  trades off how reasonable/smooth the frequency response looks vs.
                                                                    # how low we can get the f_penalty part of the objective (the part of the time-domain
                                                                    # filter outside our specified bounds).

        F_penalty1 = torch.max(torch.tensor([0], dtype=torch.float64),
                               -(F_j[1:] - F_j[:-1]) - grad_limit).sum()
        F_penalty2 = torch.max(torch.tensor([0], dtype=torch.float64),
                               (F_j[1:] - F_j[:-1])).sum()

        if iter % 100 == 0:
            print("f_penalty={}, F_penalty={}+{}".format(
                    f_penalty, F_penalty1, F_penalty2))

        O = f_penalty + F_penalty1 + F_penalty2
        return (F_j, f_k, O)


    (F_j, f_k, O) = f_and_O(0)

    print(f_k)
    print("O = {}".format(O))

    lrate = 0.001  # Learning rate
    momentum = 0.9
    momentum_grad = torch.zeros(theta_j.shape, dtype=torch.float64)

    for iter in range(4000):
        if iter % 500 == 0:
            lrate *= 0.7
        (F_j, f_k, O) = f_and_O(iter)
        if iter % 100 == 0:
            print("Iter = {}, O = {}".format(iter, O))
        O.backward()
        with torch.no_grad():
            momentum_grad = (momentum * momentum_grad) + theta_j.grad
            theta_j -= momentum_grad * lrate
            theta_j.grad.data.zero_()

        if iter in [ 2000, 3000, 3999 ]:
            plt.plot( (1.0/J)*np.arange(kN*J), f_k.detach())
            plt.plot( (math.pi/J)*np.arange(2*J + 1), F_j.detach())

    plt.ylabel('f_k, F_k')
    plt.show()



__main__()

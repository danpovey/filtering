#!/usr/bin/env python3
from itertools import count

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



D = 512   # Defines how many discrete points we use in our approximations
S = 8      # Time support of the canonical filter function would be [-S..S]... in
           # the frequency domain, almost all the energy should be in [-pi..pi].
T = 4      # how many multiples of pi we compute the freq response for


def get_fourier_matrix():
    """ Returns the matrix M_{kd} as a torch.Tensor; this maps the time
        domain filter values f_d = f(S d / D) for d=0..D-1, to
        the frequency domain gains F_k = F(k \pi / D) for k=0..(TD-1)
    """
    M = np.empty((T*D, D), dtype=np.float64)
    for d in range(D):
        if d >= 2:
            w_d = 1.0
        elif d == 0:
            w_d = 0.5
        else:
            assert d == 1
            w_d = 23.0/24.0  # see notes.txt, DIGRESSION ON APPROXIMATING
                             # INTEGRALS WITH SPLINES
        for k in range(T*D):
            c = math.sqrt(2.0 / math.pi) * (S / D) * w_d
            M[k,d] = c * math.cos(k * d * S * math.pi / (D * D))

    return torch.tensor(M)



def get_objf(F, iter, do_print=False):
    """
    This function returns the objective function.

    Args:
         F: a torch.Tensor of shape (D*T), i.e a vector.  The k'th
            element is what we call F_k in the math, which is
            F(k \pi / D) where F(omega) is the gain of the filter at
            angular frequency omega.
         iter (int):  The iteration- for diagnostics
         do_print (bool):  If true, we will print more detailed logging
            information
    Returns:
         Returns a scalar torch.Tensor that represents the loss
         function.
    """
    # This first penalty term makes sure that after taking into account
    # the contributions from the overlapping frequency bands, the total gain
    # for each frequency is 1.
    penalty1 = (((F[0:D//2]**2  + torch.flip(F[D//2:D], dims=[0])**2) -
                torch.tensor([1.0])) ** 2).sum()
    # the second penalty ensures that we have no energy outside the desired
    # frequency range.
    penalty2 = (F[D:]).abs().sum() * 2.0

    # penalty3 checks the function starts at 1
    penalty3 = (F[0] - torch.tensor([1.0])).abs().sum() * 50.0

    # penalty4 checks the function is 0 at pi
    penalty4 = F[D].abs() * 50.0

    # This is like the second penalty but starting from 2*D and with a higher
    # constant-- to strongly penalize high-freq energy.
    penalty5 = (F[2*D:]).abs().sum()

    # the third penalty ensures that we don't have negative gains,
    # which would lead to an inconvenient phase flip.
    penalty6 = torch.max(torch.tensor([0.0]), -F).sum()

    # the 7th penalty tries to ensure that F is nondecreasing, i.e.
    # the gain decreases as the frequency gets further from the origin.
    penalty7 = torch.max(torch.tensor([0.0]), F[1:] - F[:-1]).sum() * 10.0

    # penalty8 checks the function is less than 0.5 at 2.0.  Trying to
    # coax it to where I know it should go.
    #
    i = int(D * 2.0 / math.pi)
    penalty8 = torch.max(torch.tensor([0.0]), F[i] - torch.tensor([0.5])).sum() * 50.0



#         grad_limit = torch.tensor(2.5 / (2*J), dtype=torch.float64)  #  1/(2J) would be avg grad if it fell from 1 to 0 linearly.  The 2.5 is wiggle room;
#                                                                     # it  trades off how reasonable/smooth the frequency response looks vs.
#                                                                     # how low we can get the f_penalty part of the objective (the part of the time-domain
#                                                                     # filter outside our specified bounds).

#         F_penalty1 = torch.max(torch.tensor([0], dtype=torch.float64),
#                                -(F_j[1:] - F_j[:-1]) - grad_limit).sum()
#         F_penalty2 = torch.max(torch.tensor([0], dtype=torch.float64),
#                                (F_j[1:] - F_j[:-1])).sum()


    loss = penalty1 + penalty2 + penalty3 + penalty4 + penalty5 + penalty6 + penalty7 + penalty8
    if do_print:
        print("Iter {}: loss = {} = 1:{} + 2:{} + 3:{} + 4:{} + 5:{} + 6:{} + 7:{} + 8:{}".format(
                iter, loss, penalty1, penalty2, penalty3, penalty4, penalty5, penalty6, penalty7, penalty8))
    return loss

def __main__():
    torch.set_default_dtype(torch.float64)
    f = torch.ones((D), requires_grad = True)

    M = get_fourier_matrix()
    print("M = {}".format(M))

    lrate = 0.001  # Learning rate
    momentum = 0.9
    momentum_grad = torch.zeros(f.shape, dtype=torch.float64)

    for iter in range(20000):
        if iter % 1000 == 0:
            lrate *= 0.5

        F = torch.mv(M, f)
        O = get_objf(F, iter, (iter % 100 == 0))

        # Put a penalty on the second derivative of f.
        f_extended = torch.cat((torch.flip(f[1:], dims=[0]), f))
        f_deriv = f_extended[1:] - f_extended[:-1]
        f_deriv2 = f_deriv[1:] - f_deriv[:-1]
        f_penalty = torch.sqrt((f_deriv2 ** 2).sum() * 0.0001 * D * D + 0.01)

        if (iter % 100 == 0):
            print("f_penalty = {}".format(f_penalty))
        O = O + f_penalty


        O.backward()
        with torch.no_grad():
            momentum_grad = (momentum * momentum_grad) + f.grad
            f -= momentum_grad * lrate
            f.grad.data.zero_()

        if iter in [ 2000, 3000, 5000, 9999 ]:
            plt.plot( (S * 1.0 / D) * np.arange(D), f.detach())
            plt.plot( (math.pi/D) * np.arange(D*T), F.detach())

    plt.ylabel('f_k, F_k')
    plt.show()



__main__()

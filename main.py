#!/usr/bin/env python3
from itertools import count

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import filter_utils.filters as filters



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
    penalty1 = ((F[0:D//2]**2  + torch.flip(F[D//2:D], dims=[0])**2) -
                torch.tensor([1.0])).abs().sum()
    # the second penalty ensures that we have no energy outside the desired
    # frequency range.
    penalty2 = (F[D:]).abs().sum() * 2.0

    # penalty3 checks the function starts at 1
    penalty3 = (F[0] - torch.tensor([1.0])).abs().sum() * 50.0

    # This is like the second penalty but starting from 2*D and with a higher
    # constant-- to strongly penalize high-freq energy.
    penalty4 = (F[2*D:]).abs().sum()

    # the third penalty ensures that we don't have negative gains,
    # which would lead to an inconvenient phase flip.
    penalty5 = torch.max(torch.tensor([0.0]), -F).sum() * 10.0

    # the 5th penalty tries to ensure that F is nondecreasing, i.e.
    # the gain decreases as the frequency gets further from the origin.
    penalty6 = torch.max(torch.tensor([0.0]), F[1:] - F[:-1]).sum() * 10.0



#         grad_limit = torch.tensor(2.5 / (2*J), dtype=torch.float64)  #  1/(2J) would be avg grad if it fell from 1 to 0 linearly.  The 2.5 is wiggle room;
#                                                                     # it  trades off how reasonable/smooth the frequency response looks vs.
#                                                                     # how low we can get the f_penalty part of the objective (the part of the time-domain
#                                                                     # filter outside our specified bounds).

#         F_penalty1 = torch.max(torch.tensor([0], dtype=torch.float64),
#                                -(F_j[1:] - F_j[:-1]) - grad_limit).sum()
#         F_penalty2 = torch.max(torch.tensor([0], dtype=torch.float64),
#                                (F_j[1:] - F_j[:-1])).sum()


    loss = penalty1 + penalty2 + penalty3 + penalty4 + penalty5 + penalty6
    if do_print:
        print("Iter {}: loss = {} = 1:{} + 2:{} + 3:{} + 4:{} + 5:{} + 6:{}".format(
                iter, loss, penalty1, penalty2, penalty3, penalty4, penalty5, penalty6))
    return loss




def get_function_approx(x):
    scale = 1.466 # value at x=0
    first_zero_crossing = 1.72
    stddev = 2.95   # standard deviation of gaussian we multiply by
    if x == 0:
        return scale
    else:
        sinc = math.sin(x * math.pi / first_zero_crossing) * (scale / (math.pi / first_zero_crossing)) / x
        return sinc * math.exp(- x*x*(stddev ** -2))

def get_f_approx():
    x_axis = [ (S * 1.0 / D) * x for x in range(D) ]
    return torch.tensor( [ get_function_approx(x) for x in x_axis ] )


def __main__():
    torch.set_default_dtype(torch.float64)


    # We don't want the time-domain filter to have energy above an angular
    # frequency of pi, which corresponds to a frequency of 0.5.  The sampling
    # rate in the time domain (the f_t values) is D / S, so the relative
    # frequency of the cutoff is 0.5 / (D / S) = 0.5 S / D.  (This would be the
    # arg to filter_utils.filters.high_pass_filter).  This will make an
    # inconveniently wide filter, though.  We are already penalizing these high
    # energies explicitly in the fourier space, up to T * pi, so we only really
    # need to penalize in the time domain for frequencies above this; that means
    # we can boost up the relative cutoff frequency by a factor of T, giving
    # us a cutoff frequency of 0.5 S T / D.
    (f, filter_width) = filters.high_pass_filter(0.5 * S * T / D, num_zeros = 5)
    filt = torch.tensor(f)  # convert from Numpy into Torch format.


    # f_approx is a hand-tuned function very close to the 'f' we want.  The
    # optimization gets stuck in nasty local minima, and we know where we are
    # going, so we use this as a constraint.
    f_approx = get_f_approx()
    f = f_approx.clone().detach().requires_grad_(True)

    M = get_fourier_matrix()
    print("M = {}".format(M))

    lrate = 0.0001  # Learning rate
    momentum = 0.95
    momentum_grad = torch.zeros(f.shape, dtype=torch.float64)

    for iter in range(10000):
        if iter % 500 == 0:
            lrate *= 0.5

        F = torch.mv(M, f)
        O = get_objf(F, iter, (iter % 100 == 0))


        max_error = 0.01  # f should stay at least this close to f_approx.
        f_penalty1 = torch.max(torch.tensor([0.0]), torch.abs(f - f_approx) - 0.01).sum() * 5.0

        f_extended = torch.cat((torch.flip(f, dims=[0]), f))
        f_extended_highpassed = torch.nn.functional.conv1d(f_extended.unsqueeze(0).unsqueeze(0),
                                                           filt.unsqueeze(0).unsqueeze(0), padding=filter_width)
        f_extended_highpassed = f_extended_highpassed.squeeze(0).squeeze(0)
        f_penalty2 = f_extended_highpassed.abs().sum()


        if (iter % 100 == 0):
            print("f_penalty = {}+{}".format(f_penalty1, f_penalty2))
        O = O + f_penalty1 + f_penalty2

        O.backward()
        with torch.no_grad():
            momentum_grad = (momentum * momentum_grad) + f.grad
            f -= momentum_grad * lrate
            f.grad.data.zero_()

        if iter in [ 9999 ]: #[ 2000, 3000, 5000, 9999 ]:
            plt.plot( (S * 1.0 / D) * np.arange(D), f.detach())
            plt.plot( (math.pi/D) * np.arange(D*T), F.detach())

    plt.ylabel('f_k, F_k')
    plt.grid()
    plt.show()
    print("F = ", repr(F))
    print("f = ", repr(f))



__main__()

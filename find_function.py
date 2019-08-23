#!/usr/bin/env python3
from itertools import count

import math
import numpy as np

import matplotlib.pyplot as plt





D = 512   # Defines how many discrete points we use in our approximations
S = 8      # Time support of the canonical filter function would be [-S..S]... in
           # the frequency domain, almost all the energy should be in [-pi..pi].
T = 4      # how many multiples of pi we compute the freq response for


f =  [ 1.4655e+00,  1.4654e+00,  1.4650e+00,  1.4643e+00,  1.4633e+00,
         1.4619e+00,  1.4600e+00,  1.4576e+00,  1.4546e+00,  1.4509e+00,
         1.4465e+00,  1.4413e+00,  1.4355e+00,  1.4288e+00,  1.4215e+00,
         1.4136e+00,  1.4050e+00,  1.3960e+00,  1.3865e+00,  1.3767e+00,
         1.3665e+00,  1.3562e+00,  1.3457e+00,  1.3350e+00,  1.3243e+00,
         1.3134e+00,  1.3024e+00,  1.2913e+00,  1.2800e+00,  1.2685e+00,
         1.2567e+00,  1.2446e+00,  1.2322e+00,  1.2193e+00,  1.2061e+00,
         1.1924e+00,  1.1783e+00,  1.1637e+00,  1.1486e+00,  1.1332e+00,
         1.1173e+00,  1.1010e+00,  1.0844e+00,  1.0675e+00,  1.0503e+00,
         1.0329e+00,  1.0153e+00,  9.9751e-01,  9.7964e-01,  9.6170e-01,
         9.4370e-01,  9.2568e-01,  9.0764e-01,  8.8960e-01,  8.7157e-01,
         8.5356e-01,  8.3556e-01,  8.1757e-01,  7.9960e-01,  7.8162e-01,
         7.6363e-01,  7.4563e-01,  7.2760e-01,  7.0954e-01,  6.9144e-01,
         6.7330e-01,  6.5512e-01,  6.3690e-01,  6.1864e-01,  6.0036e-01,
         5.8207e-01,  5.6378e-01,  5.4551e-01,  5.2729e-01,  5.0914e-01,
         4.9108e-01,  4.7314e-01,  4.5533e-01,  4.3769e-01,  4.2024e-01,
         4.0299e-01,  3.8597e-01,  3.6919e-01,  3.5266e-01,  3.3639e-01,
         3.2038e-01,  3.0464e-01,  2.8916e-01,  2.7395e-01,  2.5900e-01,
         2.4429e-01,  2.2983e-01,  2.1561e-01,  2.0161e-01,  1.8783e-01,
         1.7427e-01,  1.6091e-01,  1.4776e-01,  1.3481e-01,  1.2206e-01,
         1.0953e-01,  9.7215e-02,  8.5124e-02,  7.3269e-02,  6.1660e-02,
         5.0312e-02,  3.9237e-02,  2.8448e-02,  1.7960e-02,  7.7844e-03,
        -2.0684e-03, -1.1589e-02, -2.0771e-02, -2.9610e-02, -3.8103e-02,
        -4.6252e-02, -5.4059e-02, -6.1531e-02, -6.8675e-02, -7.5501e-02,
        -8.2020e-02, -8.8245e-02, -9.4188e-02, -9.9863e-02, -1.0528e-01,
        -1.1046e-01, -1.1540e-01, -1.2012e-01, -1.2462e-01, -1.2891e-01,
        -1.3299e-01, -1.3687e-01, -1.4054e-01, -1.4399e-01, -1.4724e-01,
        -1.5027e-01, -1.5307e-01, -1.5565e-01, -1.5799e-01, -1.6008e-01,
        -1.6193e-01, -1.6353e-01, -1.6488e-01, -1.6598e-01, -1.6684e-01,
        -1.6745e-01, -1.6782e-01, -1.6797e-01, -1.6789e-01, -1.6762e-01,
        -1.6715e-01, -1.6651e-01, -1.6570e-01, -1.6475e-01, -1.6367e-01,
        -1.6247e-01, -1.6117e-01, -1.5977e-01, -1.5829e-01, -1.5674e-01,
        -1.5511e-01, -1.5342e-01, -1.5167e-01, -1.4985e-01, -1.4796e-01,
        -1.4600e-01, -1.4397e-01, -1.4186e-01, -1.3966e-01, -1.3738e-01,
        -1.3500e-01, -1.3253e-01, -1.2996e-01, -1.2730e-01, -1.2455e-01,
        -1.2170e-01, -1.1877e-01, -1.1577e-01, -1.1270e-01, -1.0957e-01,
        -1.0640e-01, -1.0320e-01, -9.9991e-02, -9.6775e-02, -9.3569e-02,
        -9.0385e-02, -8.7236e-02, -8.4129e-02, -8.1072e-02, -7.8071e-02,
        -7.5129e-02, -7.2247e-02, -6.9424e-02, -6.6657e-02, -6.3941e-02,
        -6.1270e-02, -5.8638e-02, -5.6035e-02, -5.3456e-02, -5.0891e-02,
        -4.8335e-02, -4.5782e-02, -4.3228e-02, -4.0670e-02, -3.8108e-02,
        -3.5543e-02, -3.2981e-02, -3.0425e-02, -2.7884e-02, -2.5367e-02,
        -2.2885e-02, -2.0448e-02, -1.8068e-02, -1.5756e-02, -1.3524e-02,
        -1.1381e-02, -9.3348e-03, -7.3922e-03, -5.5570e-03, -3.8307e-03,
        -2.2123e-03, -6.9790e-04,  7.1865e-04,  2.0459e-03,  3.2944e-03,
         4.4761e-03,  5.6039e-03,  6.6910e-03,  7.7503e-03,  8.7937e-03,
         9.8311e-03,  1.0870e-02,  1.1916e-02,  1.2971e-02,  1.4032e-02,
         1.5094e-02,  1.6149e-02,  1.7186e-02,  1.8191e-02,  1.9151e-02,
         2.0049e-02,  2.0871e-02,  2.1607e-02,  2.2244e-02,  2.2779e-02,
         2.3208e-02,  2.3535e-02,  2.3766e-02,  2.3913e-02,  2.3991e-02,
         2.4015e-02,  2.4004e-02,  2.3975e-02,  2.3942e-02,  2.3918e-02,
         2.3908e-02,  2.3902e-02,  2.3898e-02,  2.3895e-02,  2.3891e-02,
         2.3884e-02,  2.3870e-02,  2.3846e-02,  2.3808e-02,  2.3754e-02,
         2.3679e-02,  2.3581e-02,  2.3457e-02,  2.3305e-02,  2.3124e-02,
         2.2912e-02,  2.2670e-02,  2.2397e-02,  2.2096e-02,  2.1768e-02,
         2.1416e-02,  2.1042e-02,  2.0649e-02,  2.0241e-02,  1.9821e-02,
         1.9392e-02,  1.8959e-02,  1.8523e-02,  1.8088e-02,  1.7657e-02,
         1.7231e-02,  1.6811e-02,  1.6400e-02,  1.5997e-02,  1.5603e-02,
         1.5218e-02,  1.4840e-02,  1.4471e-02,  1.4107e-02,  1.3749e-02,
         1.3395e-02,  1.3043e-02,  1.2694e-02,  1.2344e-02,  1.1995e-02,
         1.1645e-02,  1.1293e-02,  1.0940e-02,  1.0585e-02,  1.0230e-02,
         9.8728e-03,  9.5158e-03,  9.1594e-03,  8.8044e-03,  8.4518e-03,
         8.1025e-03,  7.7575e-03,  7.4178e-03,  7.0841e-03,  6.7574e-03,
         6.4383e-03,  6.1274e-03,  5.8253e-03,  5.5322e-03,  5.2485e-03,
         4.9742e-03,  4.7096e-03,  4.4546e-03,  4.2090e-03,  3.9728e-03,
         3.7458e-03,  3.5277e-03,  3.3182e-03,  3.1169e-03,  2.9235e-03,
         2.7375e-03,  2.5585e-03,  2.3860e-03,  2.2193e-03,  2.0580e-03,
         1.9013e-03,  1.7488e-03,  1.5997e-03,  1.4534e-03,  1.3095e-03,
         1.1673e-03,  1.0264e-03,  8.8657e-04,  7.4753e-04,  6.0924e-04,
         4.7179e-04,  3.3544e-04,  2.0061e-04,  6.7875e-05, -6.2034e-05,
        -1.8828e-04, -3.0994e-04, -4.2603e-04, -5.3557e-04, -6.3762e-04,
        -7.3132e-04, -8.1596e-04, -8.9103e-04, -9.5622e-04, -1.0115e-03,
        -1.0571e-03, -1.0936e-03, -1.1217e-03, -1.1426e-03, -1.1577e-03,
        -1.1683e-03, -1.1763e-03, -1.1832e-03, -1.1909e-03, -1.2010e-03,
        -1.2149e-03, -1.2338e-03, -1.2588e-03, -1.2904e-03, -1.3287e-03,
        -1.3736e-03, -1.4243e-03, -1.4797e-03, -1.5385e-03, -1.5988e-03,
        -1.6588e-03, -1.7161e-03, -1.7686e-03, -1.8143e-03, -1.8512e-03,
        -1.8775e-03, -1.8921e-03, -1.8940e-03, -1.8830e-03, -1.8592e-03,
        -1.8235e-03, -1.7771e-03, -1.7219e-03, -1.6602e-03, -1.5944e-03,
        -1.5274e-03, -1.4621e-03, -1.4012e-03, -1.3474e-03, -1.3029e-03,
        -1.2695e-03, -1.2485e-03, -1.2404e-03, -1.2452e-03, -1.2621e-03,
        -1.2895e-03, -1.3255e-03, -1.3673e-03, -1.4121e-03, -1.4563e-03,
        -1.4968e-03, -1.5302e-03, -1.5535e-03, -1.5641e-03, -1.5599e-03,
        -1.5396e-03, -1.5027e-03, -1.4494e-03, -1.3808e-03, -1.2990e-03,
        -1.2066e-03, -1.1069e-03, -1.0037e-03, -9.0093e-04, -8.0282e-04,
        -7.1334e-04, -6.3611e-04, -5.7425e-04, -5.3008e-04, -5.0506e-04,
        -4.9961e-04, -5.1304e-04, -5.4355e-04, -5.8826e-04, -6.4327e-04,
        -7.0388e-04, -7.6473e-04, -8.2015e-04, -8.6441e-04, -8.9218e-04,
        -8.9895e-04, -8.8147e-04, -8.3826e-04, -7.6998e-04, -6.7970e-04,
        -5.7290e-04, -4.5724e-04, -3.4202e-04, -2.3733e-04, -1.5269e-04,
        -9.5394e-05, -6.7975e-05, -6.5249e-05, -7.0816e-05, -7.6392e-05,
        -8.2137e-05, -8.7911e-05, -9.3713e-05, -9.9511e-05, -1.0530e-04,
        -1.1114e-04, -1.1696e-04, -1.2276e-04, -1.2844e-04, -1.3411e-04,
        -1.3979e-04, -1.4535e-04, -1.4945e-04, -1.4160e-04, -1.2189e-04,
        -1.0154e-04, -8.1018e-05, -6.0474e-05, -3.9804e-05, -1.9161e-05,
         1.4711e-06,  2.2111e-05,  4.2697e-05,  6.3324e-05,  8.3919e-05,
         1.0450e-04,  1.2504e-04,  1.4540e-04,  1.6553e-04,  1.7319e-04,
         1.6501e-04,  1.5078e-04,  1.3634e-04,  1.2180e-04,  1.0723e-04,
         9.2667e-05,  7.8098e-05,  6.3501e-05,  4.8876e-05,  3.4257e-05,
         1.9721e-05,  5.1766e-06, -9.3220e-06, -2.3678e-05, -3.7091e-05,
        -3.2507e-05, -1.2092e-05,  8.6153e-06,  2.9426e-05,  5.0289e-05,
         7.1149e-05,  9.2068e-05,  1.1296e-04,  1.3389e-04,  1.5477e-04,
         1.7562e-04,  1.9622e-04];




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


def get_sinc_function(first_zero_crossing, scale, xaxis_warp, x):
    if x == 0:
        return scale
    else:
        x = x /  (1.0 + xaxis_warp * x * x * x)
        return math.sin(x * math.pi / first_zero_crossing) * (scale / (math.pi / first_zero_crossing)) / x

def __main__():
    x_axis = (S * 1.0 / D) * np.arange(D)
    plt.plot( x_axis, f)


    stddev = 2.95
    gauss_scale = np.array([ math.exp(- x*x*(stddev ** -2)) for x in x_axis])
    sinc_function = np.array([ get_sinc_function(1.715, 1.466, 0.0003, x) for x in x_axis])

    approx = gauss_scale * sinc_function

    #plt.plot(x_axis, sinc_function)
    #plt.plot(x_axis, gauss_scale)
    #plt.plot(x_axis, f / sinc_function)
    plt.plot(x_axis, approx)

    plt.plot(x_axis, approx / f)

    rel_err = np.abs(approx - f).sum() / np.abs(f).sum()
    print("Relative error is ", rel_err, ", max error is ", np.max(np.abs(approx - f)))

    plt.plot(x_axis, 100.0 * (approx - f))

    plt.ylabel('f')
    plt.grid()
    plt.show()





__main__()

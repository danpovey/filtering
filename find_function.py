#!/usr/bin/env python3
from itertools import count

import math
import numpy as np

import matplotlib.pyplot as plt





D = 512   # Defines how many discrete points we use in our approximations
S = 8      # Time support of the canonical filter function would be [-S..S]... in
           # the frequency domain, almost all the energy should be in [-pi..pi].
T = 4      # how many multiples of pi we compute the freq response for

f =   [ 1.4606e+00,  1.4603e+00,  1.4590e+00,  1.4579e+00,  1.4558e+00,
         1.4538e+00,  1.4511e+00,  1.4478e+00,  1.4443e+00,  1.4404e+00,
         1.4358e+00,  1.4309e+00,  1.4255e+00,  1.4198e+00,  1.4135e+00,
         1.4068e+00,  1.3998e+00,  1.3926e+00,  1.3846e+00,  1.3762e+00,
         1.3672e+00,  1.3583e+00,  1.3484e+00,  1.3385e+00,  1.3282e+00,
         1.3171e+00,  1.3059e+00,  1.2944e+00,  1.2828e+00,  1.2702e+00,
         1.2576e+00,  1.2445e+00,  1.2313e+00,  1.2181e+00,  1.2045e+00,
         1.1901e+00,  1.1754e+00,  1.1602e+00,  1.1452e+00,  1.1300e+00,
         1.1143e+00,  1.0985e+00,  1.0823e+00,  1.0662e+00,  1.0500e+00,
         1.0330e+00,  1.0168e+00,  9.9938e-01,  9.8247e-01,  9.6500e-01,
         9.4782e-01,  9.2991e-01,  9.1225e-01,  8.9411e-01,  8.7674e-01,
         8.5843e-01,  8.4005e-01,  8.2205e-01,  8.0446e-01,  7.8541e-01,
         7.6697e-01,  7.4850e-01,  7.3017e-01,  7.1158e-01,  6.9324e-01,
         6.7481e-01,  6.5649e-01,  6.3796e-01,  6.1967e-01,  6.0225e-01,
         5.8305e-01,  5.6506e-01,  5.4700e-01,  5.2944e-01,  5.1121e-01,
         4.9353e-01,  4.7608e-01,  4.5855e-01,  4.4218e-01,  4.2382e-01,
         4.0764e-01,  3.9032e-01,  3.7348e-01,  3.5720e-01,  3.4089e-01,
         3.2449e-01,  3.0898e-01,  2.9290e-01,  2.7772e-01,  2.6267e-01,
         2.4694e-01,  2.3166e-01,  2.1756e-01,  2.0314e-01,  1.8861e-01,
         1.7446e-01,  1.6129e-01,  1.4791e-01,  1.3413e-01,  1.2157e-01,
         1.0911e-01,  9.6637e-02,  8.4629e-02,  7.2959e-02,  6.1327e-02,
         5.0043e-02,  3.9112e-02,  2.9079e-02,  1.8661e-02,  8.1928e-03,
        -1.6813e-03, -1.0362e-02, -2.0371e-02, -2.9315e-02, -3.7961e-02,
        -4.5783e-02, -5.4208e-02, -6.1236e-02, -6.9427e-02, -7.6689e-02,
        -8.2957e-02, -9.0064e-02, -9.6615e-02, -1.0248e-01, -1.0816e-01,
        -1.1343e-01, -1.1848e-01, -1.2396e-01, -1.2858e-01, -1.3286e-01,
        -1.3703e-01, -1.4122e-01, -1.4475e-01, -1.4791e-01, -1.5091e-01,
        -1.5402e-01, -1.5637e-01, -1.5887e-01, -1.6068e-01, -1.6270e-01,
        -1.6453e-01, -1.6594e-01, -1.6650e-01, -1.6793e-01, -1.6863e-01,
        -1.6910e-01, -1.7015e-01, -1.7047e-01, -1.7043e-01, -1.7011e-01,
        -1.7005e-01, -1.6969e-01, -1.6926e-01, -1.6847e-01, -1.6753e-01,
        -1.6644e-01, -1.6520e-01, -1.6383e-01, -1.6232e-01, -1.6068e-01,
        -1.5891e-01, -1.5702e-01, -1.5477e-01, -1.5265e-01, -1.5045e-01,
        -1.4822e-01, -1.4566e-01, -1.4311e-01, -1.4048e-01, -1.3793e-01,
        -1.3547e-01, -1.3273e-01, -1.3013e-01, -1.2726e-01, -1.2434e-01,
        -1.2138e-01, -1.1838e-01, -1.1534e-01, -1.1228e-01, -1.0918e-01,
        -1.0606e-01, -1.0293e-01, -9.9772e-02, -9.6608e-02, -9.3436e-02,
        -9.0259e-02, -8.7082e-02, -8.3907e-02, -8.0739e-02, -7.7581e-02,
        -7.4436e-02, -7.1308e-02, -6.8200e-02, -6.5115e-02, -6.2056e-02,
        -5.9027e-02, -5.6030e-02, -5.3069e-02, -5.0145e-02, -4.7261e-02,
        -4.4420e-02, -4.1623e-02, -3.8873e-02, -3.6172e-02, -3.3520e-02,
        -3.0920e-02, -2.8373e-02, -2.5880e-02, -2.3441e-02, -2.1059e-02,
        -1.8734e-02, -1.6480e-02, -1.4450e-02, -1.2433e-02, -1.0244e-02,
        -8.3802e-03, -6.2444e-03, -4.2297e-03, -2.4075e-03, -5.0048e-04,
         1.2286e-03,  2.8873e-03,  4.4994e-03,  6.0399e-03,  7.5122e-03,
         8.9217e-03,  1.0269e-02,  1.1553e-02,  1.2776e-02,  1.3937e-02,
         1.5037e-02,  1.6076e-02,  1.7056e-02,  1.7940e-02,  1.8760e-02,
         1.9359e-02,  2.0233e-02,  2.0855e-02,  2.1600e-02,  2.2155e-02,
         2.2738e-02,  2.3255e-02,  2.3632e-02,  2.4071e-02,  2.4510e-02,
         2.4761e-02,  2.5071e-02,  2.5283e-02,  2.5469e-02,  2.5584e-02,
         2.5706e-02,  2.5737e-02,  2.5775e-02,  2.5776e-02,  2.5707e-02,
         2.5590e-02,  2.5512e-02,  2.5331e-02,  2.5144e-02,  2.4914e-02,
         2.4717e-02,  2.4442e-02,  2.4104e-02,  2.3908e-02,  2.3591e-02,
         2.3243e-02,  2.2906e-02,  2.2547e-02,  2.2158e-02,  2.1752e-02,
         2.1310e-02,  2.0896e-02,  2.0461e-02,  2.0003e-02,  1.9542e-02,
         1.9033e-02,  1.8586e-02,  1.8092e-02,  1.7620e-02,  1.7121e-02,
         1.6617e-02,  1.6180e-02,  1.5691e-02,  1.5198e-02,  1.4703e-02,
         1.4210e-02,  1.3719e-02,  1.3229e-02,  1.2743e-02,  1.2262e-02,
         1.1783e-02,  1.1309e-02,  1.0841e-02,  1.0378e-02,  9.9211e-03,
         9.4716e-03,  9.0295e-03,  8.5953e-03,  8.1694e-03,  7.7524e-03,
         7.3445e-03,  6.9460e-03,  6.5574e-03,  6.1788e-03,  5.8105e-03,
         5.4526e-03,  5.1054e-03,  4.7689e-03,  4.4432e-03,  4.1284e-03,
         3.8245e-03,  3.5314e-03,  3.2492e-03,  2.9777e-03,  2.7170e-03,
         2.4669e-03,  2.2274e-03,  1.9982e-03,  1.7832e-03,  1.6188e-03,
         1.4404e-03,  1.2447e-03,  1.0734e-03,  8.6929e-04,  6.9696e-04,
         5.2505e-04,  3.8386e-04,  2.5173e-04,  1.2850e-04,  1.3995e-05,
        -9.1956e-05, -1.8955e-04, -2.7898e-04, -3.6046e-04, -4.3423e-04,
        -5.0052e-04, -5.5958e-04, -6.1169e-04, -6.5711e-04, -6.9615e-04,
        -7.2909e-04, -7.5625e-04, -7.7792e-04, -7.6938e-04, -7.8689e-04,
        -7.5651e-04, -7.6854e-04, -7.7485e-04, -7.7071e-04, -7.6399e-04,
        -7.7203e-04, -7.7591e-04, -7.5889e-04, -7.3940e-04, -7.1765e-04,
        -6.9382e-04, -6.6811e-04, -6.4068e-04, -6.1170e-04, -5.8133e-04,
        -5.4971e-04, -5.1701e-04, -4.8337e-04, -4.4893e-04, -4.1385e-04,
        -3.7827e-04, -3.4235e-04, -3.0623e-04, -2.7007e-04, -2.3401e-04,
        -1.9317e-04, -1.5205e-04, -9.9419e-05, -5.8186e-05, -3.2094e-05,
        -2.0473e-05,  3.5478e-06,  3.3841e-05,  6.2910e-05,  9.0681e-05,
         1.1709e-04,  1.4208e-04,  1.6562e-04,  1.8769e-04,  2.0826e-04,
         2.2734e-04,  2.4493e-04,  2.6104e-04,  2.7569e-04,  2.8891e-04,
         3.0073e-04,  3.1117e-04,  3.2027e-04,  3.2806e-04,  3.3370e-04,
         3.3964e-04,  3.4385e-04,  3.4669e-04,  3.4834e-04,  3.4885e-04,
         3.4822e-04,  3.4647e-04,  3.4363e-04,  3.3972e-04,  3.3475e-04,
         3.2876e-04,  3.2176e-04,  3.1379e-04,  3.0488e-04,  2.9507e-04,
         2.8440e-04,  2.7524e-04,  2.6424e-04,  2.4774e-04,  2.3416e-04,
         2.1999e-04,  2.0530e-04,  1.9017e-04,  1.7466e-04,  1.5885e-04,
         1.4279e-04,  1.2656e-04,  1.1023e-04,  9.3863e-05,  7.7509e-05,
         6.1232e-05,  4.5081e-05,  2.8128e-05, -2.8736e-06, -1.2865e-05,
        -1.7398e-05, -3.2316e-05, -4.6901e-05, -6.1143e-05, -7.5027e-05,
        -8.8541e-05, -1.0168e-04, -1.1444e-04, -1.2681e-04, -1.3879e-04,
        -1.5038e-04, -1.6156e-04, -1.7232e-04, -1.8266e-04, -1.8442e-04,
        -1.9119e-04, -2.1095e-04, -2.1947e-04, -2.2745e-04, -2.3488e-04,
        -2.4176e-04, -2.4807e-04, -2.5377e-04, -2.5887e-04, -2.6335e-04,
        -2.6719e-04, -2.7040e-04, -2.7297e-04, -2.7491e-04, -2.7624e-04,
        -2.8662e-04, -2.9537e-04, -2.7670e-04, -2.7576e-04, -2.7433e-04,
        -2.7244e-04, -2.7013e-04, -2.6744e-04, -2.6439e-04, -2.6103e-04,
        -2.5739e-04, -2.5349e-04, -2.4935e-04, -2.4501e-04, -2.4048e-04,
        -2.3366e-04, -2.0258e-04, -2.0716e-04, -2.2068e-04, -2.1534e-04,
        -2.0985e-04, -2.0421e-04, -1.9842e-04, -1.9248e-04, -1.8640e-04,
        -1.8019e-04, -1.7385e-04, -1.6740e-04, -1.6085e-04, -1.5422e-04,
        -1.5629e-04, -1.9844e-04, -1.5108e-04, -1.2740e-04, -1.2075e-04,
        -1.1419e-04, -1.0773e-04, -1.0141e-04, -9.5259e-05, -8.9289e-05,
        -8.3519e-05, -7.7967e-05, -7.2640e-05, -9.0049e-06,  5.0551e-05,
        -1.2106e-05, -5.3674e-05, -4.9500e-05, -4.5538e-05, -4.1774e-05,
        -3.8196e-05, -3.8208e-04]


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


def get_function_approx(x):
    scale = 1.466 # value at x=0
    first_zero_crossing = 1.716
    stddev = 2.95   # standard deviation of gaussian we multiply by
    if x == 0:
        return scale
    else:
        sinc = math.sin(x * math.pi / first_zero_crossing) * (scale / (math.pi / first_zero_crossing)) / x
        return sinc * math.exp(- x*x*(stddev ** -2))

# The following version is more exact- about half the error, 0.004 vs. 0.008--
# but a bit less elegant.

# def get_function_approx(x):
#     scale = 1.466 # value at x=0
#     first_zero_crossing = 1.713
#     stddev = 3.1   # standard deviation of gaussian we multiply by
#     if x == 0:
#         return scale
#     else:
#         sinc = math.sin(x * math.pi / first_zero_crossing) * (scale / (math.pi / first_zero_crossing)) / x
#         return sinc * math.exp(- x*x*(stddev ** -2) - 0.1*x*x*x*x*(stddev ** -4) )

def __main__():
    x_axis = (S * 1.0 / D) * np.arange(D)
    plt.plot( x_axis, f)


    approx = np.array([ get_function_approx(x) for x in x_axis])  # gauss_scale * sinc_function

    #plt.plot(x_axis, sinc_function)
    #plt.plot(x_axis, gauss_scale)
    #plt.plot(x_axis, f / sinc_function)
    plt.plot(x_axis, approx)

    plt.plot(x_axis, approx / f)

    rel_err = np.abs(approx - f).sum() / np.abs(f).sum()
    print("Relative error is ", rel_err, ", max error is ", np.max(np.abs(approx - f)))

    plt.plot(x_axis, 100.0 * (approx - f))
    plt.ylim((-2,2))
    plt.ylabel('f')
    plt.grid()
    plt.show()





__main__()

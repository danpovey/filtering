#!/usr/bin/env python3
from itertools import count

import math
import numpy as np

import matplotlib.pyplot as plt





D = 512   # Defines how many discrete points we use in our approximations
S = 8      # Time support of the canonical filter function would be [-S..S]... in
           # the frequency domain, almost all the energy should be in [-pi..pi].
T = 4      # how many multiples of pi we compute the freq response for

f =  [ 1.4689e+00,  1.4693e+00,  1.4667e+00,  1.4652e+00,  1.4628e+00,
         1.4600e+00,  1.4568e+00,  1.4532e+00,  1.4490e+00,  1.4444e+00,
         1.4393e+00,  1.4338e+00,  1.4279e+00,  1.4215e+00,  1.4147e+00,
         1.4076e+00,  1.3999e+00,  1.3922e+00,  1.3837e+00,  1.3750e+00,
         1.3660e+00,  1.3566e+00,  1.3469e+00,  1.3368e+00,  1.3265e+00,
         1.3156e+00,  1.3046e+00,  1.2932e+00,  1.2815e+00,  1.2694e+00,
         1.2570e+00,  1.2445e+00,  1.2314e+00,  1.2178e+00,  1.2042e+00,
         1.1917e+00,  1.1777e+00,  1.1632e+00,  1.1471e+00,  1.1307e+00,
         1.1150e+00,  1.0990e+00,  1.0830e+00,  1.0667e+00,  1.0501e+00,
         1.0334e+00,  1.0164e+00,  9.9936e-01,  9.8202e-01,  9.6457e-01,
         9.4706e-01,  9.2937e-01,  9.1147e-01,  8.9360e-01,  8.7555e-01,
         8.5752e-01,  8.3945e-01,  8.2124e-01,  8.0324e-01,  7.8471e-01,
         7.6645e-01,  7.4810e-01,  7.2976e-01,  7.1161e-01,  6.9311e-01,
         6.7530e-01,  6.5681e-01,  6.3827e-01,  6.2084e-01,  6.0544e-01,
         5.8527e-01,  5.6571e-01,  5.4740e-01,  5.2889e-01,  5.1102e-01,
         4.9326e-01,  4.7562e-01,  4.5811e-01,  4.4082e-01,  4.2354e-01,
         4.0653e-01,  3.8957e-01,  3.7284e-01,  3.5637e-01,  3.4003e-01,
         3.2397e-01,  3.0800e-01,  2.9234e-01,  2.7691e-01,  2.6159e-01,
         2.4673e-01,  2.3201e-01,  2.1731e-01,  2.0328e-01,  1.8913e-01,
         1.7498e-01,  1.6191e-01,  1.4886e-01,  1.3677e-01,  1.2386e-01,
         1.0886e-01,  9.6330e-02,  8.4225e-02,  7.2476e-02,  6.0883e-02,
         4.9639e-02,  3.8683e-02,  2.8013e-02,  1.7733e-02,  7.5502e-03,
        -2.1730e-03, -1.1706e-02, -2.0859e-02, -2.9756e-02, -3.8337e-02,
        -4.6534e-02, -5.4328e-02, -6.1835e-02, -6.9224e-02, -7.6162e-02,
        -8.2556e-02, -8.9711e-02, -9.5681e-02, -1.0182e-01, -1.0727e-01,
        -1.1231e-01, -1.1862e-01, -1.2397e-01, -1.2845e-01, -1.3315e-01,
        -1.3723e-01, -1.4114e-01, -1.4476e-01, -1.4811e-01, -1.5120e-01,
        -1.5407e-01, -1.5669e-01, -1.5906e-01, -1.6116e-01, -1.6309e-01,
        -1.6471e-01, -1.6606e-01, -1.6700e-01, -1.6828e-01, -1.6895e-01,
        -1.6938e-01, -1.6942e-01, -1.6958e-01, -1.6969e-01, -1.7005e-01,
        -1.6970e-01, -1.6938e-01, -1.6891e-01, -1.6832e-01, -1.6737e-01,
        -1.6627e-01, -1.6502e-01, -1.6364e-01, -1.6213e-01, -1.6050e-01,
        -1.5874e-01, -1.5687e-01, -1.5488e-01, -1.5279e-01, -1.5058e-01,
        -1.4828e-01, -1.4579e-01, -1.4330e-01, -1.4053e-01, -1.3781e-01,
        -1.3506e-01, -1.3234e-01, -1.2971e-01, -1.2716e-01, -1.2425e-01,
        -1.2128e-01, -1.1826e-01, -1.1522e-01, -1.1214e-01, -1.0903e-01,
        -1.0590e-01, -1.0276e-01, -9.9596e-02, -9.6425e-02, -9.3248e-02,
        -9.0069e-02, -8.6892e-02, -8.3719e-02, -8.0555e-02, -7.7402e-02,
        -7.4264e-02, -7.1144e-02, -6.8045e-02, -6.4970e-02, -6.1921e-02,
        -5.8901e-02, -5.5913e-02, -5.2960e-02, -5.0043e-02, -4.7165e-02,
        -4.4329e-02, -4.1535e-02, -3.8787e-02, -3.6087e-02, -3.3435e-02,
        -3.0833e-02, -2.8284e-02, -2.5788e-02, -2.3347e-02, -2.1024e-02,
        -1.8863e-02, -1.6663e-02, -1.4487e-02, -1.2176e-02, -1.0139e-02,
        -7.9511e-03, -5.9457e-03, -4.0412e-03, -2.1916e-03, -4.0074e-04,
         1.3230e-03,  2.9755e-03,  4.5700e-03,  6.0971e-03,  7.5611e-03,
         8.9625e-03,  1.0301e-02,  1.1578e-02,  1.2791e-02,  1.3949e-02,
         1.4997e-02,  1.6009e-02,  1.6800e-02,  1.7767e-02,  1.8453e-02,
         1.9391e-02,  2.0178e-02,  2.0910e-02,  2.1685e-02,  2.2263e-02,
         2.2839e-02,  2.3312e-02,  2.3751e-02,  2.4166e-02,  2.4508e-02,
         2.4816e-02,  2.5069e-02,  2.5287e-02,  2.5460e-02,  2.5593e-02,
         2.5681e-02,  2.5734e-02,  2.5742e-02,  2.5735e-02,  2.5612e-02,
         2.5491e-02,  2.5232e-02,  2.5192e-02,  2.4978e-02,  2.4754e-02,
         2.4632e-02,  2.4413e-02,  2.4166e-02,  2.3851e-02,  2.3555e-02,
         2.3214e-02,  2.2855e-02,  2.2467e-02,  2.2089e-02,  2.1686e-02,
         2.1268e-02,  2.0835e-02,  2.0388e-02,  1.9936e-02,  1.9474e-02,
         1.8999e-02,  1.8513e-02,  1.8037e-02,  1.7514e-02,  1.6995e-02,
         1.6545e-02,  1.6104e-02,  1.5606e-02,  1.5125e-02,  1.4629e-02,
         1.4134e-02,  1.3641e-02,  1.3150e-02,  1.2663e-02,  1.2179e-02,
         1.1699e-02,  1.1225e-02,  1.0757e-02,  1.0295e-02,  9.8391e-03,
         9.3910e-03,  8.9507e-03,  8.5186e-03,  8.0951e-03,  7.6805e-03,
         7.2751e-03,  6.8793e-03,  6.4932e-03,  6.1171e-03,  5.7511e-03,
         5.3955e-03,  5.0502e-03,  4.7155e-03,  4.3914e-03,  4.0779e-03,
         3.7750e-03,  3.4829e-03,  3.2014e-03,  2.9306e-03,  2.6704e-03,
         2.4207e-03,  2.1816e-03,  1.9991e-03,  1.8030e-03,  1.5782e-03,
         1.3701e-03,  1.1790e-03,  9.9141e-04,  8.0434e-04,  6.3683e-04,
         4.8805e-04,  3.4865e-04,  2.1847e-04,  9.7316e-05, -1.5002e-05,
        -1.1869e-04, -2.1398e-04, -3.0109e-04, -3.8028e-04, -4.5181e-04,
        -5.1594e-04, -5.7296e-04, -6.2316e-04, -6.6684e-04, -7.0429e-04,
        -7.1854e-04, -7.2234e-04, -7.3662e-04, -7.6240e-04, -7.7606e-04,
        -7.8650e-04, -7.8522e-04, -7.8466e-04, -7.9003e-04, -7.9177e-04,
        -7.9031e-04, -7.7556e-04, -7.5807e-04, -7.3804e-04, -7.1566e-04,
        -6.9110e-04, -6.6456e-04, -6.3620e-04, -6.0619e-04, -5.7471e-04,
        -5.4192e-04, -5.0799e-04, -4.7310e-04, -4.3742e-04, -4.0112e-04,
        -3.6437e-04, -3.2735e-04, -2.9022e-04, -2.3691e-04, -1.8069e-04,
        -1.6083e-04, -1.2029e-04, -8.6591e-05, -6.1801e-05, -4.0271e-05,
        -8.4433e-06,  2.2862e-05,  5.2950e-05,  8.1752e-05,  1.0922e-04,
         1.3530e-04,  1.5997e-04,  1.8321e-04,  2.0501e-04,  2.2536e-04,
         2.4428e-04,  2.6178e-04,  2.7788e-04,  2.9258e-04,  3.0591e-04,
         3.1790e-04,  3.2856e-04,  3.3790e-04,  3.4596e-04,  3.5274e-04,
         3.5825e-04,  3.6251e-04,  3.6553e-04,  3.6732e-04,  3.6789e-04,
         3.6726e-04,  3.6543e-04,  3.6242e-04,  3.5826e-04,  3.5802e-04,
         3.5168e-04,  3.3911e-04,  3.3062e-04,  3.2115e-04,  3.1075e-04,
         2.9946e-04,  2.8736e-04,  2.7451e-04,  2.6097e-04,  2.4682e-04,
         2.3212e-04,  2.1696e-04,  2.0139e-04,  1.8550e-04,  1.6936e-04,
         1.5303e-04,  1.3658e-04,  1.2008e-04,  1.0356e-04,  7.3046e-05,
         5.8655e-05,  5.4494e-05,  3.8441e-05,  2.2592e-05,  6.9828e-06,
        -8.3712e-06, -2.3445e-05, -3.8226e-05, -5.2701e-05, -6.6858e-05,
        -8.0691e-05, -9.4189e-05, -1.0734e-04, -1.2015e-04, -1.3258e-04,
        -1.4464e-04, -1.5630e-04, -1.5667e-04, -1.7644e-04, -1.8875e-04,
        -1.9864e-04, -2.0803e-04, -2.1689e-04, -2.2520e-04, -2.3292e-04,
        -2.4004e-04, -2.4654e-04, -2.5240e-04, -2.5761e-04, -2.6216e-04,
        -2.6604e-04, -2.6926e-04, -2.7183e-04, -2.7601e-04, -2.9364e-04,
        -2.7727e-04, -2.7591e-04, -2.7552e-04, -2.7462e-04, -2.7325e-04,
        -2.7146e-04, -2.6927e-04, -2.6672e-04, -2.6385e-04, -2.6069e-04,
        -2.5727e-04, -2.5359e-04, -2.4970e-04, -2.4559e-04, -2.4129e-04,
        -2.0502e-04, -2.2186e-04, -2.2727e-04, -2.2223e-04, -2.1700e-04,
        -2.1160e-04, -2.0600e-04, -2.0023e-04, -1.9427e-04, -1.8815e-04,
        -1.8186e-04, -1.7543e-04, -1.6887e-04, -1.6220e-04, -1.5544e-04,
        -1.8504e-04, -1.7802e-04, -1.3496e-04, -1.2817e-04, -1.2145e-04,
        -1.1483e-04, -1.0834e-04, -1.0201e-04, -9.5861e-05, -8.9917e-05,
        -8.4191e-05, -7.8694e-05, -7.3437e-05, -4.4527e-06,  5.3106e-05,
        -3.4399e-05, -5.4744e-05, -5.0630e-05, -4.6708e-05, -4.2968e-05,
        -3.9398e-05, -3.8312e-04]



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
    first_zero_crossing = 1.72
    stddev = 2.95   # standard deviation of gaussian we multiply by
    if x == 0:
        return scale
    else:
        sinc = math.sin(x * math.pi / first_zero_crossing) * (scale / (math.pi / first_zero_crossing)) / x
        return sinc * math.exp(- x*x*(stddev ** -2))

def __main__():
    x_axis = (S * 1.0 / D) * np.arange(D)
    plt.plot( x_axis, f)


    stddev = 2.95
    #gauss_scale = np.array([ math.exp(- x*x*(stddev ** -2)) for x in x_axis])
    #sinc_function = np.array([ get_sinc_function(1.72, 1.466, 0.00, x) for x in x_axis])

    approx = np.array([ get_function_approx(x) for x in x_axis])  # gauss_scale * sinc_function

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

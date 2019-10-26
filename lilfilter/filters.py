# To be run with python3

"""
This module contains functions that compute and apply filters (currently just
FIR filters, both causal and non-causal).  A filter, as we define it here, is a
tuple

  (f, i)

where f is a numpy.ndarray with one axis, and dtype=float32, float64, complex64
or complex128, containing filter coefficients; and i is an integer in the range
[0 .. f.shape[0]-1] which represents time t=0 of the filter.  The direction of
the coefficients is as for an impulse response; and we will convolve the input
signal with the filter.  When applied to the signal x_t, the output y_t will be:

  y_t = \sum_s a_s x_{t+s}

where a_s is the filter coefficient for time s, defined physically as f[s-i],
and the summation over s is taken over all s where that expression would be
valid (excluding negative array indexes).

We truncate the output as dictated by the 'i' coefficient (the time t=0 index
into the array f); the behavior we are looking for is that if f is all zeros
except f[i] being one, then applying the filter should leave the signal
unchanged.
"""


import numpy as np
import math


def check_is_filter(filt):
    """
    This function raises an exception if `filt` is not a filter (see the top of this
    file for explanation.  A filter is defined here as a tuple (f, i) where f is
    an np.ndarray with one axis (i.e. f.ndim == 1), i (which represents the
    index of the array f that corresponds to t == 0), satisfies 0 <= i <
    f.shape[0]; f must also have dtype=float32 or complex64.
    """
    if not isinstance(filt, tuple):
        raise TypeError("Expected filt to be a filter, got type {}".format(type(filt)))
    if not len(filt) == 2:
        raise ValueError("Expected input to be a filter which is a 2-tuple, got a "
                         "{}-tuple".format(len(filt)))
    if not isinstance(filt[0], np.ndarray):
        raise TypeError("Expected f to be a filter but 1st element has "
                        "type {}".format(type(filt[0])))
    if not filt[0].dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        raise TypeError("Expected f to be a filter but array has dtype={}".format(
                filt[0].dtype))
    if not isinstance(filt[1], int):
        raise TypeError("Expected f to be a filter but 2nd element has "
                        "type {}".format(type(filt[1])))
    if not (len(filt[0].shape) == 1 and filt[1] >= 0 and filt[1] < filt[0].shape[0]):
        raise ValueError("Shape and central-index mismatch: f.shape={}, i={}".format(
                filt[0].shape, filt[1]))


def check_is_signal(sig, axis):
    """
    Checks that `sig` is a signal, with `axis` in the correct range.
    This means that `sig` is an np.ndarray with dtype in
    [np.float32, np.float64, np.complex64, np.complex128],
    and axis is in [-sig.ndim .. sig.ndim-1].
    Raise an exception if it is not a signal.
    """
    if not isinstance(sig, np.ndarray):
        raise TypeError("Expected signal to be of type numpy.ndarray, got "
                        "{}".format(type(sig)))
    if not sig.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        raise TypeError("Expected dtype of signal to be of a float or complex type, "
                        "got {}".format(sig.dtype))
    if axis < -sig.ndim or axis >= sig.ndim:
        raise TypeError("axis out of range: was {} but ndim is {}".format(
                axis, sig.ndim))

def apply_filter(filt, signal, axis=-1, out=None):
    """
    Applies the filter `filt` to the signal `signal`, returning the result with
    the same dimension as `signal`.
    `filt` is a filter as defined at the top of this file; `signal`
    is a numpy.ndarray containing a signal or signals; `axis` is the time axis of
    the signal;(-1 means the last axis).
    """
    check_is_filter(filt)
    check_is_signal(signal, axis)

    (f,i) = filt

    ndim = signal.ndim
    (filt_len,) = f.shape

    if out is None:
        out = np.empty(signal.shape, dtype=signal.dtype)
    else:
        if out.shape != signal.shape:
            raise ValueError("Expected `out` to have same shape as `signal`, {} != {}".format(
                    out.shape, signal.shape))

    if axis != -1 and axis != signal.ndim - 1:
        signal.swapaxes(axis, -1)
        out.swapaxes(axis, -1)

    def apply(input, output):
        if input.ndim == 1:
            temp = np.convolve(input, f)
            print("i = ", i, ", filt_len = ", filt_len, ", output shape = ", output.shape,
                  ", temp shape = ", temp.shape)
            output[:] = temp[i:temp.shape[0] - (filt_len-i-1)]
        else:
            for j in range(input.shape[0]):
                apply(input[j,:], output[j,:])

    apply(signal, out)
    return out



def gaussian_filter(stddev, stddevs_cutoff = 4.0):
    """
    Returns a symmetric Gaussian filter (low pass, obviously)

    Args:
    stddev (float):  The standard deviation of the Gaussian
       filter; this has the dimension of a time in samples.
       Must be greater than 1 (and probably considerably more
       than 1), or it wouldn't make sense asthere would be
       too much aliasing.

    stddevs_cutoff (float):  Determines the width of
       support of the filter, which will be stddev times
       this value.
    """
    assert stddev > 1.0 and stddevs_cutoff > 1.0
    neg_inv_var = -1.0 / (stddev * stddev)
    i = int(stddev * stddevs_cutoff)
    f = np.empty(2*i + 1)
    for n in range(2*i + 1):
        t = n - i
        f[n] = math.exp(t*t*neg_inv_var)
    # Normalize directly: simpler than getting the constant right.
    f *= 1.0 / f.sum()
    ans = (f,i)
    check_is_filter(ans)
    return ans


def low_pass_filter(cutoff, num_zeros = 16):
    """
    Returns a low-pass non-causal FIR filter with no phase shift.

    Args:
       cutoff (float):
          Frequency cutoff of the filter relative to the sampling frequency
          of the signal, i.e. if sample frequency is S and filter cutoff is
          F, cutoff = F / S.  Must satisfy 0 < cutoff <= 0.5 (note:
          0.5 is the Nyquist)
      num_zeros (int):
          We truncate the filter at this-numbered zero of the sinc function, and
          window with a raised-cosine function that goes to zero at
          this-numbered zero.  Larger value means sharper cutoff and wider
          filter.  Must be >= 2.
    """
    assert isinstance(cutoff, float) and cutoff > 0 and cutoff <= 0.5
    assert isinstance(num_zeros, int) and num_zeros >= 2

    # The canonical filter function would be, if C == cutoff,
    #    f(t) = 2C sinc(2Ct),
    # where `sinc` is the normalized sinc function, sinc(t) = sin(pi t) / (pi
    # t).
    # To make this finite we multiply it by a raised-cosine window
    # g(t), and use g(t) f(t) as the filter.  The raised-cosine window
    # f(t) will have support on [-z/2C, z/2C]  where z == num_zeros.

    C = cutoff
    samples_each_side = math.ceil(num_zeros / (2.0 * C))

    def sinc_function(t):
        """ Returns f(t) = 2C sinc(2Ct) where sinc is the normalized
            sinc function, sinc(t) = sin(pi t) / (pi t)
            t is the time in samples relative to the t=0 of the filter,
            may be positive or negative; expected to be an integer.
            """
        if t == 0:
            return  2.0 * C
        else:
            # 2 * C *  math.sin(2 * math.pi * C * t) / (2 * pi * C * t)
            # simplifies to the following:
            return  math.sin(2 * math.pi * C * t) / (math.pi * t)

    def cosine_window(t):
        # The window will be nonzero on [-w..w].
        #
        w = num_zeros / (2.0 * C)
        if t <= -w or t >= w:
            return 0.0
        else:
            # map the interval [-w..w] to [-pi..pi] before taking cosine.
            return 0.5 + 0.5 * math.cos(t * math.pi / w)

    out = np.empty(samples_each_side * 2 + 1)

    for t in range(-samples_each_side, samples_each_side + 1):
        out[t + samples_each_side] = sinc_function(t) * cosine_window(t)

    return (out, samples_each_side)


def high_pass_filter(cutoff, num_zeros = 5):
    """
    Returns a low-pass non-causal FIR filter with no phase shift.
    See the documentation for `low_pass_filter` for more info.
    This is just the unit filter (i.e. the filter that does nothing)
    minus that.
    """
    (f, i) = low_pass_filter(cutoff, num_zeros)
    # Negate the filter function and add 1.0 to its central position.
    f = -f
    f[i] += 1.0
    return (f, i)


def hilbert_filter(n = 300):
    """
    Returns a filter that applies a truncated version of the Hilbert
    transform, i.e. the filter is 1/t times a raised-cosine window
    with support on [-n .. n].

    This will
    """

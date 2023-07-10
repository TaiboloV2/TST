import numpy as np

def movingAverage(x, N):
    # convolve with average filter
    out = np.convolve(x, np.ones((N)) / N, mode="same")
    # fix edges
    offset = np.arange(int(np.ceil(N/2)), N, 1)
    index = N // 2
    # start
    out[:index] = out[:index] * N / offset
    # end
    out[-index:] = out[-index:] * N / np.flip(offset)
    return out
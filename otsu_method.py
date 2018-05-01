import numpy as np
from collections import Counter

def Thresholding_Otsu(img):
    nbins = 256
    pixel_counts = Counter(img.ravel())
    counts = np.array([0 for x in range(256)])
    for c in sorted(pixel_counts):
        counts[c] = pixel_counts[c]
        p = counts / sum(counts)
        sigma_b = np.zeros((256, 1))
    for t in range(nbins):
        q_L = sum(p[:t])
        q_H = sum(p[t:])
        if q_L == 0 or q_H == 0:
            continue

        miu_L = sum(np.dot(p[:t], np.transpose(np.matrix([i for i in range(t)])))) / q_L
        miu_H = sum(np.dot(p[t:], np.transpose(np.matrix([i for i in range(t, 256)])))) / q_H
        sigma_b[t] = q_L * q_H * (miu_L - miu_H) ** 2

    return np.argmax(sigma_b)

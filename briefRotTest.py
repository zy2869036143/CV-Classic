import numpy as np
import cv2
import scipy.ndimage
from matchPics import matchPics
from helper import plotMatches
import matplotlib.pyplot as plt

img = cv2.imread("./images/cv_cover.jpg")
match_count = np.zeros(36)
bins = np.arange(0, 360, 10)
for i in range(36):
    # Rotate Image
    rotated_img = scipy.ndimage.rotate(img, i * 10)
    # Compute features, descriptors and Match features
    match, locs1, locs2 = matchPics(img, rotated_img)
    # plotMatches(img, rotated_img, match, locs1, locs2)
    match_count[i] = match.shape[0]
    print("%d finished." % (i + 1))
    # Update histogram
plt.bar(bins, match_count, width=30)
plt.grid(True,linestyle=':',color='peru',alpha=0.6)
plt.xlabel('rotation')
plt.ylabel('matches')
plt.show()

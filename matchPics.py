import cv2
from helper import briefMatch, computeBrief, fast_corner_detection, plotMatches

"""
I1: Images to match, BGR pattern
I2: Images to match, BGR pattern
"""
def matchPics(I1, I2):
    # I1, I2 : Images to match
    # Convert Images to GrayScale
    img_gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    # Detect Features in Both Images
    locs1 = fast_corner_detection(img_gray1)
    locs2 = fast_corner_detection(img_gray2)
    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(img_gray1, locs1)
    desc2, locs2 = computeBrief(img_gray2, locs2)
    # Match features using the descriptors
    matches = briefMatch(desc1, desc2)
    return matches, locs1, locs2
if __name__ == "__main__":
    images = {
        "ML1": "images/ML1.jpg",
        "ML2": "images/ML2.jpg"
    }
    img1 = cv2.imread(images["ML1"])
    img2 = cv2.imread(images["ML2"])
    matches, locs1, locs2 = matchPics(img1, img2)
    plotMatches(img1, img2, matches, locs1, locs2)
import numpy as np
import copy
import cv2


class HarrisCornerDetector:
    def __init__(self, kernel_size=3, sigma=2):
        self.sobel_x = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.sobel_y = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
        self.kernel_size = kernel_size
        self.gaussian_function = lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        self.gaussian_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for x in range(kernel_size):
            for y in range(kernel_size):
                round_x = x - kernel_size // 2
                round_y = y - kernel_size // 2
                self.gaussian_filter[y, x] = self.gaussian_function(round_x, round_y)
        self.gaussian_filter /= (sigma * np.sqrt(2 * np.pi))
        self.gaussian_filter /= self.gaussian_filter.sum()

    def __bgr_to_gray__(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img

    def __hessian__(self, gray_img):
        Ix = np.zeros_like(gray_img, dtype=np.float32)
        Iy = np.zeros_like(gray_img, dtype=np.float32)
        h, w = gray_img.shape
        padded_img = np.pad(gray_img, (1, 1), 'edge')
        for y in range(h):
            for x in range(w):
                Ix[y, x] = np.mean(padded_img[y:y + 3, x:x + 3] * self.sobel_x)
                Iy[y, x] = np.mean(padded_img[y:y + 3, x:x + 3] * self.sobel_y)
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy
        return Ix2, Iy2, Ixy

    def __gaussian_filter__(self, img):
        h, w = img.shape
        padded_img = np.pad(img, (self.kernel_size // 2, self.kernel_size // 2), 'edge')
        result = np.zeros_like(img, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                result[y, x] = np.sum(padded_img[y:y + self.kernel_size, x:x + self.kernel_size] * self.gaussian_filter)
        return result

    def __corner_detect__(self, img, ix2, iy2, ixy, k, th):
        out = copy.deepcopy(img)
        # Compute R
        R = (ix2 * iy2 - ixy ** 2) - k * ((ix2 + iy2) ** 2)
        # Detect corners that satisfy the condition.
        out[R >= np.max(R) * th] = [0, 0, 255]
        out = out.astype(np.uint8)
        return out

    def __corner_cordinate__(self, gray_img, ix2, iy2, ixy, k, th):
        h, w = gray_img.shape
        R = (ix2 * iy2 - ixy ** 2) - k * ((ix2 + iy2) ** 2)
        key_points = []
        threshold = np.max(R) * th
        for y in range(h):
            for x in range(w):
                if R[y, x] >=  threshold:
                    key_points.append([x, y])
        return key_points

    def detect(self, img, k=0.04, threshold=0.05):
        # Step1: Convert bgr image to gray image
        self.gray_img = self.__bgr_to_gray__(img)
        # Step2: Compute hessian matrix.
        Ix2, Iy2, Ixy = self.__hessian__(self.gray_img)
        # Step3: Apply gaussian filter to the hessian matrix.
        Ix2 = self.__gaussian_filter__(Ix2)
        Iy2 = self.__gaussian_filter__(Iy2)
        Ixy = self.__gaussian_filter__(Ixy)
        # Detect Corner with threshold and k.
        result_img = self.__corner_detect__(img, Ix2, Iy2, Ixy, k, threshold)
        return result_img

    def get_key_points(self, img, k=0.04, threshold=0.05):
        # Step1: Convert bgr image to gray image
        gray_img = self.__bgr_to_gray__(img)
        # Step2: Compute hessian matrix.
        Ix2, Iy2, Ixy = self.__hessian__(gray_img)
        # Step3: Apply gaussian filter to the hessian matrix.
        Ix2 = self.__gaussian_filter__(Ix2)
        Iy2 = self.__gaussian_filter__(Iy2)
        Ixy = self.__gaussian_filter__(Ixy)
        # Detect Corner with threshold and k.
        key_points = self.__corner_cordinate__(gray_img, Ix2, Iy2, Ixy, k, threshold)
        return key_points

if __name__ == "__main__":
    # usage example
    img = cv2.imread("./images/cv_cover.jpg")
    harris_detector = HarrisCornerDetector()
    result = harris_detector.detect(img)
    cv2.imshow("Corner Result", result)
    cv2.waitKey(0)

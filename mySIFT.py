import numpy as np
import cv2
import scipy.ndimage
import copy
from myHarris import HarrisCornerDetector
import matplotlib.pyplot as plt


def under_sample(image, step):
    result = image[::step, ::step]
    return result


def gaussian_kernel(kernel_size, sigma):
    x_range = [i - (kernel_size // 2) for i in range(kernel_size)]
    assistant = []
    for i in range(kernel_size):
        assistant.append(x_range)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    kernel = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    return kernel


def convolve(kernel, img, padding, strides):
    result = None
    kernel_size = kernel.shape
    img_size = img.shape
    if len(img_size) == 3:
        channel = []
        for i in range(img_size[-1]):
            pad_img = np.pad(img[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            temp = []
            for j in range(0, img_size[0], strides[1]):
                temp.append([])
                for k in range(0, img_size[1], strides[0]):
                    val = (kernel * pad_img[j * strides[1]:j * strides[1] + kernel_size[0],
                                    k * strides[0]:k * strides[0] + kernel_size[1]]).sum()
                    temp[-1].append(val)
            channel.append(np.array(temp))
        channel = tuple(channel)
        result = np.dstack(channel)
    elif len(img_size) == 2:
        channel = []
        pad_img = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
        for j in range(0, img_size[0], strides[1]):
            channel.append([])
            for k in range(0, img_size[1], strides[0]):
                val = (kernel * pad_img[j * strides[1]:j * strides[1] + kernel_size[0],
                                k * strides[0]:k * strides[0] + kernel_size[1]]).sum()
                channel[-1].append(val)
        result = np.array(channel)
    return result


class SIFT():
    count = 0

    @staticmethod
    def draw_key_points(image, key_points):
        image = copy.deepcopy(image)
        for point in key_points:
            # point: [x_in_origin, y_in_origin, x_in_DoG, y_in_DoG, octave, layer, scale, radius, degree]
            y = int(point[1])
            x = int(point[0])
            radis = int(np.round(point[7]))
            cv2.circle(image, (x, y), radis, (0, 0, 255))
            endy = int(y + radis * np.sin(np.deg2rad(point[8])))
            endx = int(x + radis * np.cos(np.deg2rad(point[8])))
            cv2.line(image, (x, y), (endx, endy), (0, 0, 255))
        cv2.imshow("key points and directions %d" % (SIFT.count), image)
        SIFT.count += 1

    @staticmethod
    def match_key_points(key_points_list1, descriptors_1, key_points_list2, descriptors_2, max_distance=40000):
        indexes = []
        deltas = []
        for i in range(len(key_points_list1)):
            dsc1 = descriptors_1[i]
            min_distance = max_distance
            target = -1
            for j in range(len(key_points_list2)):
                dsc2 = descriptors_2[j]
                distance_vector = np.array(dsc1) - np.array(dsc2)
                distance = distance_vector.dot(distance_vector)
                if distance <= min_distance:
                    min_distance = distance
                    target = j
            if target != -1:
                indexes.append([i, target])
                deltas.append(min_distance)
        return indexes, deltas

    @staticmethod
    def draw_match_result(image1, key_points_list1, image2, key_points_list2, indexes):
        h1, w1, _ = image1.shape
        h2, w2, _ = image2.shape
        padded_image1 = image1
        padded_image2 = image2

        if h1 < h2:
            padded_image1 = np.pad(image1, ((0, h2 - h1), (0, 0), (0, 0)))
        elif h1 > h2:
            padded_image2 = np.pad(image2, ((0, h1 - h2), (0, 0), (0, 0)))

        if w1 < w2:
            padded_image1 = np.pad(padded_image1, ((0, 0), (0, w2 - w1), (0, 0)))
        elif w1 > w2:
            padded_image2 = np.pad(padded_image2, ((0, 0), (0, w1 - w1), (0, 0)))

        image_combine = np.hstack((padded_image1, padded_image2))

        for index in indexes:
            point1 = key_points_list1[index[0]]
            point2 = key_points_list2[index[1]]
            _, w, _ = padded_image2.shape
            cv2.line(image_combine, (int(point1[0]), int(point1[1])), (int(point2[0]) + w, int(point2[1])), (0, 0, 255))

        cv2.imshow("Match Result", image_combine)

    def __init__(self, image, sigma=1.6, n=2):
        self.n = n
        self.k = 2 ** (1 / n)
        self.SIGMA = sigma
        self.INIT_SIGMA = 0.5
        # 1.52
        self.sigma0 = np.sqrt(self.SIGMA ** 2 - self.INIT_SIGMA ** 2)
        self.original_image = image
        self.gray_image = self.__to_gray__(image)
        h, w = self.gray_image.shape
        self.MAX_OCTAVE_NUM = int(np.log2(np.mean([h, w]))) - 3
        self.LAYER_IN_OCTAVE = n + 3
        self.gaussian_pyramid_sigmas = [
            [((self.k ** n) ** octave) * (self.k ** i) * self.sigma0 for i in range(self.LAYER_IN_OCTAVE)]
            for octave in range(self.MAX_OCTAVE_NUM)]
        # Empty gaussian pyramid
        self.gaussian_pyramid = [[] for _ in range(self.MAX_OCTAVE_NUM)]
        self.dog_pyramid = None
        """
            In the following codes, we locate pixels in a image in the following way:
                image[y, x]
             → x        
           ↓ [[                  ]
           y  [        image     ]            
              [                  ]]
        """

    def __to_gray__(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_gray

    def __get_DoG__(self, show=False):
        octave0 = [under_sample(self.gray_image, octave * 2 if octave > 0 else 1) for octave in
                   range(self.MAX_OCTAVE_NUM)]
        for octave in range(self.MAX_OCTAVE_NUM):
            for pic in range(self.LAYER_IN_OCTAVE):
                kernel_size = int(6 * self.gaussian_pyramid_sigmas[octave][pic] + 1)
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
                kernel = gaussian_kernel(kernel_size, self.gaussian_pyramid_sigmas[octave][pic])
                conv_result = convolve(kernel, octave0[octave],
                                       [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2], [1, 1])
                self.gaussian_pyramid[octave].append(conv_result)
        self.dog_pyramid = [
            [self.gaussian_pyramid[octave][pic + 1] - self.gaussian_pyramid[octave][pic] for pic in
             range(self.LAYER_IN_OCTAVE - 1)]
            for octave in range(self.MAX_OCTAVE_NUM)]
        if show:
            plt.figure(1)
            for i in range(self.MAX_OCTAVE_NUM):
                for j in range(self.n + 3):
                    array = np.array(self.gaussian_pyramid[i][j], dtype=np.float32)
                    plt.subplot(self.MAX_OCTAVE_NUM, self.n + 3, j + (i) * self.MAX_OCTAVE_NUM + 1)
                    plt.imshow(array.astype(np.uint8), cmap='gray')
                    plt.axis('off')
            plt.show()

            plt.figure(2)
            for i in range(self.MAX_OCTAVE_NUM):
                for j in range(self.n + 2):
                    array = np.array(self.dog_pyramid[i][j], dtype=np.float32)
                    plt.subplot(self.MAX_OCTAVE_NUM, self.n + 3, j + (i) * self.MAX_OCTAVE_NUM + 1)
                    plt.imshow(array.astype(np.uint8), cmap='gray')
                    plt.axis('off')
            plt.show()
        return self.gaussian_pyramid, self.dog_pyramid

    def __is_3d_extreme__(self, octave, layer, y, x, SIFT_FIXPT_SCALE=1, contrastThreshold=0.04):
        threshold = 0.5 * contrastThreshold / (self.n * 255 * SIFT_FIXPT_SCALE)
        pixel_value = self.dog_pyramid[octave][layer][y, x]
        # Threshold
        if abs(pixel_value) < threshold:
            return False
        h, w = self.dog_pyramid[octave][layer].shape
        # Avoid crossing the image boundary.
        if (y - 1) < 0:
            y = 1
        elif (y + 2) > h:
            y = h - 2
        if (x - 1) < 0:
            x = 1
        elif (x + 2) > w:
            x = w - 2
        # Find extremes in its 26 neighbours
        if pixel_value >= 0:
            if pixel_value < np.max(self.dog_pyramid[octave][layer][y - 1:y + 2, x - 1:x + 2]):
                return False
            if pixel_value < np.max(self.dog_pyramid[octave][layer + 1][y - 1:y + 2, x - 1:x + 2]):
                return False
            if pixel_value < np.max(self.dog_pyramid[octave][layer - 1][y - 1:y + 2, x - 1:x + 2]):
                return False
        else:
            if pixel_value > np.min(self.dog_pyramid[octave][layer][y - 1:y + 2, x - 1:x + 2]):
                return False
            if pixel_value > np.min(self.dog_pyramid[octave][layer + 1][y - 1:y + 2, x - 1:x + 2]):
                return False
            if pixel_value > np.min(self.dog_pyramid[octave][layer - 1][y - 1:y + 2, x - 1:x + 2]):
                return False
        return True

    def __compute_accurate_location__(self, octave, layer, y, x):
        MAX_ITERATION_STEPS = 5
        IMAGE_BORDER = 5
        currnet_step = 0
        derive_scale = 255 * 2
        second_derive_scale = 255
        cross_derive_scale = 255 * 4
        aD_aX = np.zeros((3, 1), dtype=np.float64)
        a2D_aX2 = np.zeros((3, 3), dtype=np.float64)
        X = np.zeros(3, dtype=int)
        X[0] = x
        X[1] = y
        X[2] = layer
        image = self.dog_pyramid[octave][layer]

        def cross_boundary(X):
            if X[2] < 1 or X[2] > self.n or X[1] < IMAGE_BORDER or X[1] >= image.shape[0] - IMAGE_BORDER or X[
                0] < IMAGE_BORDER or X[0] >= image.shape[1] - IMAGE_BORDER:
                return True
            return False

        while currnet_step < MAX_ITERATION_STEPS:
            if cross_boundary(X):
                return None
            x = X[0]
            y = X[1]
            layer = X[2]

            image = self.dog_pyramid[octave][layer]
            up_image = self.dog_pyramid[octave][layer + 1]
            down_image = self.dog_pyramid[octave][layer - 1]

            """
            有限差分法求导:
            aD/aX = [aD/ax, aD/ay, aD/aσ].T
            a2D/aX2 = [ [a2D/(ax*ax), a2D/(ax*ay), a2D/(ax*aσ)],
                        [a2D/(ax*ay), a2D/(ay*ay), a2D/(ay*aσ)]
                        [a2D/(ax*aσ), a2D/(ay*aσ), a2D/(aσ*aσ)] ]
            """
            aD_aX[0] = (image[y, x + 1] - image[y, x - 1]) / derive_scale
            aD_aX[1] = (image[y + 1, x] - image[y - 1, x]) / derive_scale
            aD_aX[2] = (up_image[y, x] - down_image[y, x]) / derive_scale

            a2D_aX2[1, 1] = (image[y + 1, x] + image[y - 1, x] - 2 * image[y, x]) / second_derive_scale
            a2D_aX2[0, 0] = (image[y, x + 1] + image[y, x - 1] - 2 * image[y, x]) / second_derive_scale
            a2D_aX2[2, 2] = (up_image[y, x] + down_image[y, x] - 2 * image[y, x]) / second_derive_scale
            a2D_aX2[0, 1] = a2D_aX2[1, 0] = (image[y + 1, x + 1] + image[y - 1, x - 1] - image[y + 1, x - 1] - image[
                y - 1, x + 1]) / cross_derive_scale
            a2D_aX2[0, 2] = a2D_aX2[2, 0] = (up_image[y, x + 1] + down_image[y, x - 1] - up_image[y, x - 1] -
                                             down_image[y, x + 1]) / cross_derive_scale
            a2D_aX2[1, 2] = a2D_aX2[2, 1] = (up_image[y + 1, x] + down_image[y - 1, x] - up_image[y - 1, x] -
                                             down_image[y + 1, x]) / cross_derive_scale

            delta_X = -np.matmul(np.linalg.pinv(a2D_aX2), aD_aX)

            if np.max(abs(delta_X)) < 0.5:
                break

            round = np.round(delta_X[:, 0]).astype(int)
            X += round
            currnet_step += 1

        if currnet_step == MAX_ITERATION_STEPS:
            return None
        if cross_boundary(X):
            return None

        # Low contract point has a high probability of being a noisy point.
        # We must eliminate them by the following code block.
        t = aD_aX.dot(delta_X.T)[0, 0]
        image_scale = 255
        contrastThreshold = 0.04
        contract = image[X[1], X[0]] / image_scale + t * 0.5
        if abs(contract) * self.n < contrastThreshold:
            return None

        # Use hessian Matrix to avoid edge effect with a threshold of r=10
        r = 10.0
        hessian = a2D_aX2[0:2, 0:2]
        TrH = hessian[0, 0] + hessian[1, 1]
        DetH = hessian[0, 0] * hessian[1, 1] - hessian[1, 0] * hessian[0, 1]
        if DetH < 0 or TrH ** 2 * r >= (r + 1) ** 2 * DetH:
            return None

        # Congratulations! The point finally passed all examinations!
        append_X = np.zeros(7, dtype=np.float32)
        append_X[0] = (X[0] + delta_X[0]) * 2 ** octave  # real x in original image
        append_X[1] = (X[1] + delta_X[1]) * 2 ** octave  # real y in original image
        append_X[2] = X[0]  # x in DoG image
        append_X[3] = X[1]  # y in Dog image
        append_X[4] = octave  # octave
        append_X[5] = X[2]  # layer
        append_X[6] = self.SIGMA * np.power(2.0, (X[2] + delta_X[2]) / self.n) * 2 ** (octave + 1)  # scale
        return append_X

    def __main_direction__(self, X, SIFT_ORI_PEAK_RATIO=0.8):
        """
        :param X: [x_in_origin, y_in_origin, x_in_DoG, y_in_DoG, octave, layer, scale]
        :return: X: [x_in_origin, y_in_origin, x_in_DoG, y_in_DoG, octave, layer, scale, radius, degree]
        """
        SIFT_ORI_RADIUS = 3 * self.sigma0

        scl_octave = X[6] * 0.5 / 2 ** X[4]
        exp_scale = -1.0 / (2.0 * (self.sigma0 * scl_octave) ** 2)
        radius = int(np.round(SIFT_ORI_RADIUS * scl_octave))
        image = self.gaussian_pyramid[int(X[4])][int(X[5])]

        dxs = []
        dys = []
        degrees = []
        weight = []

        x = int(X[2])
        y = int(X[3])

        DIRECTION_SAMPLES = 36
        for delta_y in range(-radius, radius + 1):
            # Avoid crossing the boundary
            if y + delta_y - 1 < 0 or y + delta_y + 1 >= image.shape[0]:
                continue
            for delta_x in range(-radius, radius + 1):
                # Avoid crossing the boundary
                if x + delta_x - 1 < 0 or x + delta_x + 1 >= image.shape[1]:
                    continue
                # Compute derivartive
                dx = image[y + delta_y, x + delta_x + 1] - image[y + delta_y, x + delta_x - 1]
                dy = image[y + delta_y - 1, x + delta_x] - image[y + delta_y + 1, x + delta_x]
                # Attention please: In the following line degree ∈[-Π/2， Π/2]
                degree = np.rad2deg(np.arctan2(dy, dx))
                # Convert degree in order that degree∈[0, 2Π).
                if degree < 0:
                    degree += 360.0
                # Save the
                dxs.append(dx)
                dys.append(dy)
                degrees.append(degree)
                weight.append((delta_x ** 2 + delta_y ** 2) * exp_scale)

        statistics = np.zeros((DIRECTION_SAMPLES,), dtype=np.float64)
        dxs = np.array(dxs)
        dys = np.array(dys)
        degrees = np.array(degrees)
        weight = np.exp(np.array(weight))
        # mag = (dxs ** 2 + dys ** 2) ** 0.5
        length = np.sqrt(dxs ** 2 + dys ** 2) * weight
        # length2 = mag * weight
        for index, degree in enumerate(degrees):
            statistics[int(np.round(degree / 10)) % 36] += length[index]

        temp = [statistics[DIRECTION_SAMPLES - 1], statistics[DIRECTION_SAMPLES - 2], statistics[0], statistics[1]]

        statistics = np.insert(statistics, 0, temp[0])
        statistics = np.insert(statistics, 0, temp[1])
        statistics = np.insert(statistics, len(statistics), temp[2])
        statistics = np.insert(statistics, len(statistics), temp[3])

        final_hist = np.zeros((DIRECTION_SAMPLES,), dtype=np.float64)
        for i in range(DIRECTION_SAMPLES):
            final_hist[i] = (
                    (statistics[i] + statistics[i + 4]) * (1.0 / 16.0) + (statistics[i + 1] + statistics[i + 3]) * (
                        4.0 / 16.0) +
                    statistics[i + 2] * (6.0 / 16.0))

        max_index = np.argmax(final_hist)
        max_magnitude = final_hist[max_index]
        point_list = []
        for i in range(DIRECTION_SAMPLES):
            # Peak and larger than 0.8 max magnitude.
            if final_hist[i - 1] < final_hist[i] > final_hist[(i + 1) % DIRECTION_SAMPLES] and final_hist[
                i] > max_magnitude * SIFT_ORI_PEAK_RATIO:
                """
                    value:    a         m       b 
                                        |
                              |         |   
                              |         |       |
                              |         |       |
                    index:   i-1        i      i+1   ---> index increase direction
                    the following line computes: max_index + 0.5 * (a-b)/(a-2*m+b)
                """
                bin = i + 0.5 * (final_hist[i - 1] - final_hist[(i + 1) % 36]) / (
                            final_hist[i - 1] - 2 * final_hist[i] + final_hist[(i + 1) % 36])
                if bin < 0:
                    bin += DIRECTION_SAMPLES
                elif bin >= DIRECTION_SAMPLES:
                    bin %= DIRECTION_SAMPLES
                point = np.zeros(9, dtype=np.float64)
                point[0:7] = X
                point[7] = radius
                point[8] = bin * 360.0 / DIRECTION_SAMPLES
                point_list.append(point)

        return point_list

    def get_key_points(self):
        key_points = []
        for octave in range(self.MAX_OCTAVE_NUM):
            for layer in range(1, self.LAYER_IN_OCTAVE - 2):
                h, w = self.dog_pyramid[octave][layer].shape
                for y in range(h):
                    for x in range(w):
                        if not self.__is_3d_extreme__(octave, layer, y, x):
                            continue
                        point = self.__compute_accurate_location__(octave, layer, y, x)
                        if point is None:
                            continue
                        point_append_list = self.__main_direction__(point)
                        for point in point_append_list:
                            key_points.append(point)
        key_points = np.array(key_points)
        return key_points

    def get_key_points_harris(self):
        harris_detector = HarrisCornerDetector()
        key_points_init = harris_detector.get_key_points(self.original_image)
        key_points_final = []
        for point in key_points_init:
            # point_detail = self.__compute_accurate_location__(0, 0, point[1], point[0])
            point_detail = np.zeros(7, dtype=np.float32)
            point_detail[2] = point_detail[0] = point[0]
            point_detail[3] = point_detail[1] = point[1]
            point_detail[6] = 5
            # if point_detail is None:
            #     continue
            point_append_list = self.__main_direction__(point_detail)
            for point_detail in point_append_list:
                key_points_final.append(point_detail)
        return key_points_final

    def __single_point_descriptor__(self, image, point_y_x, ori, scale, d, n, SIFT_DESCR_SCL_FCTR=3.0,
                                    SIFT_DESCR_MAG_THR=0.2,
                                    SIFT_INT_DESCR_FCTR=512.0, FLT_EPSILON=1.2E-07):
        h, w = image.shape
        hist_width = SIFT_DESCR_SCL_FCTR * scale

        dst = []
        point = [int(np.round(point_y_x[0])), int(np.round(point_y_x[1]))]

        cos_t = np.cos(np.deg2rad(ori))
        sin_t = np.sin(np.deg2rad(ori))

        cos_t /= hist_width
        sin_t /= hist_width

        bins_per_rad = 360.0 / n
        exp_scale = -1.0 / (d * d * 0.5)
        # Radius might be larger than the actual sample region. But that doesn't matter.
        radius = int(np.round(hist_width * (d + 1) * 0.5))

        hist = np.zeros((d + 2) * (d + 2) * (n + 2), dtype=np.float64)
        linear_3d_index = lambda i, j, k: ((i * (d + 2)) + j) * (n + 2) + k

        X = []
        Y = []
        x_bins = []
        y_bins = []
        # Bin = []
        # CBin = []
        W = []

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):

                x_rot = j * cos_t - i * sin_t
                y_rot = j * sin_t + i * cos_t

                x_bin = x_rot + d // 2 - 0.5
                y_bin = y_rot + d // 2 - 0.5

                y = point[1] + i
                x = point[0] + j

                if x_bin > -1 and x_bin < d and y_bin > -1 and y_bin < d and y > 0 and y < h - 1 and x > 0 and x < w - 1:
                    dx = (image[y, x + 1] - image[y, x - 1])
                    dy = (image[y - 1, x] - image[y + 1, x])
                    X.append(dx)
                    Y.append(dy)
                    x_bins.append(x_bin)
                    y_bins.append(y_bin)
                    W.append((x_rot * x_rot + y_rot * y_rot) * exp_scale)

        length = len(W)
        Y = np.array(Y)
        X = np.array(X)
        orientation = np.rad2deg(np.arctan2(Y, X))
        rotated_orientation = (orientation - ori) / bins_per_rad
        Mag = np.sqrt(X ** 2 + Y ** 2)
        W = np.exp(np.array(W))

        for k in range(length):
            xbin = x_bins[k]
            ybin = y_bins[k]
            obin = rotated_orientation[k]
            magnitude = Mag[k] * W[k]

            x = int(xbin)
            y = int(ybin)
            o = int(obin)
            xbin -= x
            ybin -= y
            obin -= o

            if o < 0:
                o += n
            if o >= n:
                o -= n

            # Tri-linear Interpolation
            hist[linear_3d_index(x + 1, y + 1, o)] += magnitude * (1 - xbin) * (1 - ybin) * (1 - obin)
            hist[linear_3d_index(x + 1, y + 1, o + 1)] += magnitude * (1 - xbin) * (1 - ybin) * obin
            hist[linear_3d_index(x + 1, y + 2, o)] += magnitude * (1 - xbin) * ybin * (1 - obin)
            hist[linear_3d_index(x + 1, y + 2, o + 1)] += magnitude * (1 - xbin) * ybin * obin
            hist[linear_3d_index(x + 2, y + 1, o)] += magnitude * xbin * (1 - ybin) * (1 - obin)
            hist[linear_3d_index(x + 2, y + 1, o + 1)] += magnitude * xbin * (1 - ybin) * obin
            hist[linear_3d_index(x + 2, y + 2, o)] += magnitude * xbin * ybin * (1 - obin)
            hist[linear_3d_index(x + 2, y + 2, o + 1)] += magnitude * xbin * ybin * obin

        # finalize histogram, since the orientation histograms are circular
        for i in range(d):
            for j in range(d):
                idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
                hist[idx] += hist[idx + n]
                hist[idx + 1] += hist[idx + n + 1]
                for k in range(n):
                    dst.append(hist[idx + k])

        # Normalization
        dst = np.array(dst, dtype=np.float64)
        threshold = np.sqrt(np.sum(dst ** 2)) * SIFT_DESCR_MAG_THR
        dst[dst > threshold] = threshold
        nrm = SIFT_INT_DESCR_FCTR / max(np.sqrt(np.sum(dst ** 2)), FLT_EPSILON)
        dst = dst * nrm
        dst[dst < 0] = 0
        dst[dst > 255] = 255

        return dst

    def get_descriptors(self, gpyr, keypoints, SIFT_DESCR_WIDTH=4, SIFT_DESCR_HIST_BINS=8):
        # SIFT_DESCR_WIDTH = 4，描述直方图的宽度
        # SIFT_DESCR_HIST_BINS = 8 描述直方图的宽度
        descriptors = []
        # [x_in_origin, y_in_origin, x_in_DoG, y_in_DoG, octave, layer, scale, radius, degree]
        for i in range(len(keypoints)):
            kpt = keypoints[i]
            octave = int(kpt[4])
            layer = int(kpt[5])
            scale = 1.0 / 2 ** octave  # 缩放倍数
            size = kpt[6] * scale  # 该特征点所在组的图像尺寸
            point_y_x = [kpt[0] * scale, kpt[1] * scale]
            image = gpyr[octave][layer]
            descriptors.append(self.__single_point_descriptor__(image, point_y_x, kpt[8], size * 0.5, SIFT_DESCR_WIDTH,
                                                                SIFT_DESCR_HIST_BINS))
        descriptors = np.array(descriptors)
        return descriptors

    def detect_and_describe(self, shwo_dog=False, harris=False):
        gaussian_pyramid, DoG_pyramid = self.__get_DoG__(shwo_dog)
        key_points = None
        if harris:
            key_points = self.get_key_points_harris()
        else:
            key_points = self.get_key_points()
        descriptors = self.get_descriptors(gaussian_pyramid, key_points)
        return key_points, descriptors


if __name__ == "__main__":
    path = "images/cv_cover.jpg"
    path2 = "./images/house.jpg"
    path3 = "./images/me_s.jpg"
    path4 = "./images/me_side_s.jpg"

    image = cv2.imread(path3)
    image_rot = cv2.imread(path4)

    # image_rot = scipy.ndimage.rotate(image, 180)

    sift1 = SIFT(image)
    key_points1, descriptors1 = sift1.detect_and_describe(harris=False)

    sift2 = SIFT(image_rot)
    key_points2, descriptors2 = sift2.detect_and_describe(harris=False)

    SIFT.draw_key_points(image, key_points1)
    SIFT.draw_key_points(image_rot, key_points2)

    indexes, distances = SIFT.match_key_points(key_points1, descriptors1, key_points2, descriptors2, max_distance=70000)
    SIFT.draw_match_result(image, key_points1, image_rot, key_points2, indexes)
    cv2.waitKey(0)

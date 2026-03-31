import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage

def gaussian_smooth(size, sigma=1):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Gaussian Smoothing                                     #
    #   Input: window size, sigma                                          #
    #   Output: smoothing image                                            #
    ########################################################################

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    normal = 1.0 / (2.0 * np.pi * sigma**2)
    img = normal * np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    img = img / np.sum(img)

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return img


def sobel_edge_detection(im, sigma):
    ########################################################################
    # TODO:                                                                #
    #   Perform the sobel edge detection                                   #
    #   Input: image after smoothing                                       #
    #   Output: the magnitude and direction of gradient                    #
    ########################################################################

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    grad_x = ndimage.convolve(im.astype(np.float32), sobel_x, mode='reflect')
    grad_y = ndimage.convolve(im.astype(np.float32),  sobel_y, mode='reflect')

    gradient_magnitude = np.hypot(grad_x, grad_y)
    gradient_direction = np.arctan2(grad_y, grad_x)

    threshold = 0.1 * np.max(gradient_magnitude)
    gradient_magnitude[gradient_magnitude < threshold] = 0
    gradient_direction[gradient_magnitude == 0] = 0

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return  (gradient_magnitude, gradient_direction)


def structure_tensor(gradient_magnitude, gradient_direction, k, sigma):
    ########################################################################
    # TODO:                                                                #
    #   Perform the cornermess response                                    #
    #   Input: gradient_magnitude, gradient_direction                      #
    #   Output: second-moment matrix of Structure Tensor                   #
    ########################################################################

    Ix = gradient_magnitude * np.cos(gradient_direction)
    Iy = gradient_magnitude * np.sin(gradient_direction)

    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    Sxx = ndimage.gaussian_filter(Ixx, sigma)
    Sxy = ndimage.gaussian_filter(Ixy, sigma)
    Syy = ndimage.gaussian_filter(Iyy, sigma)

    det_M = (Sxx * Syy)
    trace_M = Sxx + Syy
    StructureTensor = det_M - k * (trace_M ** 2)

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return  StructureTensor

def NMS(harrisim,window_size=30,threshold=0.1):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Non-Maximum Suppression                                #
    #   Input: Structure Tensor, window size, threshold                    #
    #   Output: filtered coordinators                                      #
    ########################################################################

    corner_threshold = threshold * harrisim.max()
    harrisim_t = harrisim.copy()
    harrisim_t[harrisim_t < corner_threshold] = 0

    data_max = ndimage.maximum_filter(harrisim_t, size=window_size, mode='constant')
    maxima = (harrisim_t == data_max)
    maxima[harrisim_t == 0] = False

    coords = np.array(np.nonzero(maxima)).T
    candidate_values = harrisim_t[maxima]
    index = np.argsort(candidate_values)[::-1]

    filtered_coords = []
    allowed_locations = np.zeros(harrisim.shape, dtype=bool)
    half = window_size // 2
    allowed_locations[half:-half, half:-half] = True

    for i in index:
        y, x = coords[i]
        if allowed_locations[y, x]:
            filtered_coords.append((y, x))
            y0 = max(0, y - half)
            y1 = min(harrisim.shape[0], y + half + 1)
            x0 = max(0, x - half)
            x1 = min(harrisim.shape[1], x + half + 1)
            allowed_locations[y0:y1, x0:x1] = False

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return filtered_coords

def plot_harris_points(image,filtered_coords):

    plt.figure()
    plt.gray()
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0]for p in filtered_coords],'+')
    plt.axis('off')
    plt.show()
    
def rotate(image, angle, center = None, scale = 1.0):

    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
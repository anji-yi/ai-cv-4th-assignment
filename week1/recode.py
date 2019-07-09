import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import os


def read_image(filepath):
    return cv2.imread(filepath)


def read_image_grayscale(filepath):
    return cv2.imread(filepath, 0)


def show_image(img):
    cv2.imshow("lenna", img)


def crop_image(img, ratio):
    """Crop the image to a certain ratio."""
    return img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)]


def color_split(img, color):
    """Split BGR color channels and return the specified channel.
    
    :param img: input image
    :param color: 'b', 'g', or 'r'
    :return: the specified color channel (blue, green, or red)
    """

    color_idx = {"b": 0, "g": 1, "r": 2}
    return img[:, :, color_idx[color]]


def random_light_color(img, delta):
    """Randomly change pixel values in each color channel."""

    color_changed = []
    for color in ["b", "g", "r"]:
        img_color = color_split(img, color)
        # Generate a random number from -50 to 50
        rand = random.randint(-delta, delta)
        # Add the random number to the color channel
        img_color = img_color + rand
        # Set the value limits to 0 - 255
        img_color[img_color > 255] = 255
        img_color[img_color < 0] = 0

        color_changed.append(img_color.astype(img.dtype))

    # Merge channels and return the changed image
    return cv2.merge(tuple(color_changed))


def gamma_correction(img, gamma):
    """Gamma correction: O = I ^ (1 / G)"""

    invGamma = 1.0 / gamma
    # Create a lookup table to map pixel values [0,255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )

    # Apply gamma correction on the input image using the lookup table
    return cv2.LUT(img, table)


def histogram_equalization(img):
    """Stretch the pixel value histogram to either ends. 
       Increase the contrast of the image."""

    # Convert image to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Equalize the histrogram of the Y channel
    # Y: luminance, UV: chrominance
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # Convert the YUV image back to RGB
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def rotation(img, degree):
    rows, cols, ch = img.shape
    # Rotate, keep the same center, without scaling
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)

    return cv2.warpAffine(img, M, (cols, rows))


def affine_transformation(img):
    """All parallel lines in the original image will still be parallel in the output image. 
        To find the transformation matrix, we need three points from input image and their corresponding locations in output image. 
        Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine."""

    rows, cols, ch = img.shape

    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32(
        [[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]]
    )

    M = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(img, M, (cols, rows))


def perspective_transformation(img, delta):
    """For perspective transformation, you need a 3x3 transformation matrix. 
        Straight lines will remain straight even after the transformation. 
        To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. 
        Among these 4 points, 3 of them should not be collinear. 
        Then transformation matrix can be found by the function cv2.getPerspectiveTransform. 
        Then apply cv2.warpPerspective with this 3x3 transformation matrix."""

    h, w, _ = img.shape

    rand1 = random.randint(-delta, delta)
    rand2 = random.randint(-delta, delta)

    # Select a random point to transform
    rand_pt = random.randint(0, 3)
    pts1 = [[0,0], [w-1,0], [w-1,h-1], [0,h-1]]
    pts2 = pts1.copy()
    pts2[rand_pt] = [pts2[rand_pt][0] + rand1, pts2[rand_pt][1] + rand2]

    M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))

    return cv2.warpPerspective(img, M, (w, h))


if __name__ == "__main__":

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(proj_dir), "assets")
    filepath = os.path.join(assets_dir, "lenna.jpg")

    # Read image
    img = read_image(filepath)

    # Read image in gray scale
    img_gray = read_image_grayscale(filepath)

    # Show image
    show_image(img)

    # Show gray scale image
    show_image(img_gray)

    # Show cropped image
    img_crop = crop_image(img, 0.5)
    show_image(img_crop)

    # Show split color
    img_blue = color_split(img, "b")
    show_image(img_blue)

    # Random change color
    img_changed = random_light_color(img, 50)
    show_image(img_changed)

    # Gamma correction
    gamma = 1.5
    img_brighter = gamma_correction(img, gamma)
    show_image(img_brighter)

    # Histogram equalization
    img_eq = histogram_equalization(img_brighter)
    show_image(img_eq)

    # Rotation
    img_rot = rotation(img, 90)
    show_image(img_rot)

    # Affine transformation
    img_aff = affine_transformation(img)
    show_image(img_aff)

    # Perspective transformation
    img_persp = perspective_transformation(img, 200)
    show_image(img_persp)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

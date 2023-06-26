# -*- coding: utf-8 -*-
"""
Functions to process images of sketch maps and detect markings on them
"""
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance
from skimage.exposure import match_histograms


def detect_markings(
    sketch_map_markings: NDArray,
    colour: str,
    threshold_bgr: float = 0.5,
) -> NDArray:
    """
    Detect areas in a specified colour in an image containing only markings on a sketch map. I.e. all non-marked
    areas need to be set to zero (black) before calling this function.

    :param sketch_map_markings: Image containing all marked areas of a sketch map, with all other pixels set to zero.
    :param colour: Colour the markings in which should be detected.
                   Possible values: 'white', 'red', 'blue', 'green', 'yellow', 'turquoise', 'pink'
    :param threshold_bgr: Threshold for the colour detection. 0.5 means 50%, i.e. all BGR values above 50% * 255 will be
                          considered 255, all values below this threshold will be considered 0 for determining the
                          colour of the markings.
    :return: Image with all pixels marked in the specified colour set to 255 and all others set to zero.
    """
    threshold_bgr_abs = threshold_bgr * 255

    colours = {
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "turquoise": (255, 255, 0),
        "pink": (255, 0, 255),
    }
    bgr = colours[colour]

    # for color, bgr in colors.items():
    single_colour_marking = np.zeros_like(sketch_map_markings, np.uint8)
    single_colour_marking[
        (
            (sketch_map_markings[:, :, 0] < threshold_bgr_abs)
            == (bgr[0] < threshold_bgr_abs)
        )
        & (
            (sketch_map_markings[:, :, 1] < threshold_bgr_abs)
            == (bgr[1] < threshold_bgr_abs)
        )
        & (
            (sketch_map_markings[:, :, 2] < threshold_bgr_abs)
            == (bgr[2] < threshold_bgr_abs)
        )
    ] = 255
    single_colour_marking = _reduce_noise(single_colour_marking)
    single_colour_marking = _reduce_holes(single_colour_marking)
    single_colour_marking[single_colour_marking > 0] = 255
    return single_colour_marking


def _reduce_noise_(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)
    img = cv2.divide(gray, blurred, scale=255)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # TODO: Remove noise from regions where globes are placed (e.g. x% width and height rectangles)
    return cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 20, 21)


def detect_markings_(
    sketch_map_markings: NDArray,
    colour: Tuple[int, int, int],
) -> NDArray:
    range_interval = 20
    sketch_map_markings_hsv = cv2.cvtColor(sketch_map_markings, cv2.COLOR_BGR2HSV)
    single_colour_marking = cv2.inRange(sketch_map_markings_hsv, np.array(colour)- range_interval, np.array(colour) + range_interval)
    single_colour_marking = cv2.cvtColor(single_colour_marking, cv2.COLOR_GRAY2BGR)
    single_colour_marking = _reduce_noise_(single_colour_marking)
    single_colour_marking = _reduce_holes(single_colour_marking)
    single_colour_marking[single_colour_marking > 0] = 255
    return single_colour_marking


def prepare_img_for_marking_detection(
    img_base: NDArray,
    img_markings: NDArray
) -> NDArray:
    """
    Based on an image of a sketch map with markings and another image of the same sketch map without markings,
    retain the areas of the image of the marked sketch map which contain markings for further processing, set
    the unmarked areas to black.

    :img_base: Image of the unmarked sketch map.
    :img_markings: Image of the marked sketch map.
    :return: Image containing only the marked areas of a sketch map, all other pixels set to zero.
    """
    img_base_height, img_base_width, _ = img_base.shape
    img_markings = cv2.resize(
        img_markings,
        (img_base_width, img_base_height),
        fx=4,
        fy=4,
        interpolation=cv2.INTER_NEAREST,
    )
    img_markings_gray = cv2.cvtColor(img_markings, cv2.COLOR_BGR2GRAY)
    img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    img_markings_contrast = _enhance_contrast(img_markings_gray)
    img_diff = cv2.absdiff(img_base_gray, img_markings_contrast)
    smoothed = cv2.GaussianBlur(img_diff, (5, 5), 0)
    _, mask_markings = cv2.threshold(smoothed, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_markings = np.array(mask_markings, dtype=bool)
    markings_multicolor = np.zeros_like(img_markings, np.uint8)
    markings_multicolor[mask_markings] = img_markings[mask_markings]
    return markings_multicolor


def _enhance_contrast(img: NDArray, factor: float = 2.0) -> NDArray:
    """
    Enhance the contrast of a given image

    :param img: Image of which the contrast should be enhanced.
    :param factor: Factor for the contrast enhancement.
    :return: Image with enhanced contrast.
    """
    input_img = Image.fromarray(img)
    return np.array(ImageEnhance.Contrast(input_img).enhance(factor))


def _reduce_noise(img: NDArray, factor: int = 2) -> NDArray:
    """
    Reduce the noise, i.e. artifacts, in an image containing markings

    :param img: Image in which the noise should be reduced.
    :param factor: Kernel size (x*x) for the noise reduction.
    :return: 'img' with less noise.
    """
    # See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    reduced_noise = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, np.ones((factor, factor), np.uint8)
    )
    # TODO: Long running job in next line -> Does the slightly improved noise
    #       reduction justify keeping it?
    return cv2.fastNlMeansDenoisingColored(reduced_noise, None, 30, 30, 20, 21)


def _reduce_holes(img: NDArray, factor: int = 4) -> NDArray:
    """
    Reduce the holes in markings on a given image

    :param img: Image in which the holes should be reduced.
    :param factor: Kernel size (x*x) of the reduction.
    :return: Image with fewer and smaller 'holes'.
    """
    # See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((factor, factor), np.uint8))


def reduce_nr_of_colours(img: NDArray, max_nr: int = 30) -> Tuple[NDArray, NDArray]:
    """
    Reduce the number of colours in an image by performing colour quantisation.
    C.f. https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

    :param img: Image which should be returned with fewer colours. Should have 3 channels (BGR).
    :param max_nr: Max. number of colours in the returned image.
    :return: Image with fewer different colours (BGR), Remaining colours (BGR colour codes).
    """
    img_array = np.float32(img.reshape((img.shape[0] * img.shape[1], 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, colours = cv2.kmeans(img_array, max_nr, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    colours = np.uint8(colours)
    img_with_fewer_colours_array = colours[labels.flatten()]
    img_with_fewer_colours = img_with_fewer_colours_array.reshape(img.shape)
    return img_with_fewer_colours, colours


def equalise_colours(img: NDArray, reference: NDArray) -> NDArray:
    """
    Equalise the colours of a given image with those of a second given image to account, for example, for
    different light conditions during taking a photo.
    C.f. https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html

    :param img: Image the colours of which are to be adjusted based on a second image.
    :param reference: Image based on which the colours of the first image are adjusted.
    :return: First image 'img' with equalised colours.
    """
    return match_histograms(img, reference, channel_axis=-1)


if __name__ == "__main__":
    # Note: HSV in GIMP 0-360, 0-100, 0-100. OpenCV: 0-179, 0-255, 0-255
    def gimp_hsv_to_opencv_hsv(a, b, c):
        return 179*a/360, 255*b/100, 255*c/100

    # Manually detected using GIMP for the first example
    colours_hsv_gimp = [(336, 80, 75, "red"), (217, 69, 68, "blue"), (152, 85, 62, "green"), (290, 8, 30, "black")]

    # Save relevant results with CTRL+S for comparisons
    test_cases = (
        ("tests/fixtures/marking-detection/scan-base-map.jpg", "tests/fixtures/marking-detection/scan-markings.jpg"),
       # ("tests/fixtures/marking-detection/photo-base-map.jpg", "tests/fixtures/marking-detection/photo-markings.jpg")
    )
    for case in test_cases:
        img_base = cv2.imread(case[0])
        img_markings = cv2.imread(case[1])

        img_markings_eq = equalise_colours(img_markings, img_base)
        img_base_red, centers_base = reduce_nr_of_colours(img_base)
        img_markings_red, centers_markings = reduce_nr_of_colours(img_markings_eq)

        # TODO: Compare colour distributions to detect marking colours

        result = prepare_img_for_marking_detection(img_base, img_markings)
        cv2.imshow("Result of 'prepare_img_for_marking_detection'", result)
        cv2.waitKey(0)

        for col in colours_hsv_gimp:
            result_single_col = detect_markings_(result, gimp_hsv_to_opencv_hsv(*col[:3]))
            cv2.imwrite(f"new_noise_method_{col[3]}.jpg", result_single_col)
            if col[3] != "black":
                result_single_col = detect_markings(result, col[3])
                cv2.imwrite(f"old_method_{col[3]}.jpg", result_single_col)
            # cv2.imshow(f"Result of 'detect_markings' for colour", result_single_col)
            # cv2.waitKey(0)

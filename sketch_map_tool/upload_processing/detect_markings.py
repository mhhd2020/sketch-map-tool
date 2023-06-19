# -*- coding: utf-8 -*-
"""
Functions to process images of sketch maps and detect markings on them
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance


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
    img_markings_contrast = _enhance_contrast(img_markings)
    img_diff = cv2.absdiff(img_base, img_markings_contrast)
    img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
    _, mask_markings = cv2.threshold(img_diff_gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    input_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = ImageEnhance.Contrast(input_img).enhance(factor)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


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


if __name__ == "__main__":
    # Save relevant results with CTRL+S for comparisons

    test_cases = (
        ("tests/fixtures/marking-detection/scan-base-map.jpg", "tests/fixtures/marking-detection/scan-markings.jpg"),
        ("tests/fixtures/marking-detection/photo-base-map.jpg", "tests/fixtures/marking-detection/photo-markings.jpg")
    )
    for case in test_cases:
        img_base = cv2.imread(case[0])
        img_markings = cv2.imread(case[1])
        result = prepare_img_for_marking_detection(img_base, img_markings)
        cv2.imshow("Result of 'prepare_img_for_marking_detection'", result)
        cv2.waitKey(0)

        # for colour in ('white', 'red', 'blue', 'green', 'yellow', 'turquoise', 'pink'):
        #     result_single_col = detect_markings(result, colour)
        #     cv2.imshow(f"Result of 'detect_markings' for colour '{colour}'", result_single_col)
        #     cv2.waitKey(0)

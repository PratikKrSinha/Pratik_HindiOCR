import cv2
import numpy as np
from PIL import Image


def preprocess_image(image):

    img = np.array(image)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # find character
    coords = cv2.findNonZero(thresh)

    x, y, w, h = cv2.boundingRect(coords)

    cropped = thresh[y:y+h, x:x+w]

    # resize

    resized = cv2.resize(cropped,(64,64))

    return resized
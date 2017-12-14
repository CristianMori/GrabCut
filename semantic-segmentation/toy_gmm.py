import cv2
import numpy as np

foreground = cv2.imread("Transparent Training Set/Flower3FGT.png", cv2.IMREAD_UNCHANGED)
background = cv2.imread("Transparent Training Set/Flower3BGT.png", cv2.IMREAD_UNCHANGED)


def filter_and_flatten(image: np.ndarray, thresh_hold = 200):
    assert image.shape[2] == 4  # assert it has alpha channel
    ret = np.reshape(image, (-1, 4))
    ret = np.array([visible[:, :3] for visible in ret if visible[3] > thresh_hold])
    return ret







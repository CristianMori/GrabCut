import cv2
import numpy as np
import os
from functools import partial
from grabcut.cut import *
from semantic_segmentation.gmm_image import GMMImage

from sklearn import mixture


def filter_and_flatten(image: np.ndarray, thresh_hold=200):
    assert image.shape[2] == 4  # assert it has alpha channel
    ret = np.reshape(image, (-1, 4))
    ret = np.array([visible[:3] for visible in ret if visible[3] > thresh_hold])
    return ret


def get_gmm_from_folert(path_to_folder: str):
    assert os.path.isdir(path_to_folder)
    files = [os.path.join(path_to_folder, files) for files in os.listdir(path_to_folder)]
    files = map(partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED), files)
    images = map(filter_and_flatten, files)
    pixels = np.vstack(images)
    gmm = GMMImage()
    gmm.fit(pixels)
    return gmm


# fgGMM = GMM.load_gmm_from_values(*get_gmm_from_folert('semantic_segmentation/Training Set/Foreground/Car'))
# bgGMM = GMM.load_gmm_from_values(*get_gmm_from_folert('semantic_segmentation/Training Set/Background/Car'))

fgGMM = get_gmm_from_folert('semantic_segmentation/Training Set/Foreground/Car')
bgGMM = get_gmm_from_folert('semantic_segmentation/Training Set/Background/Car')

test_image = cv2.imread('semantic_segmentation/Test Images/car3.jpg')  # type: np.ndarray
probably_foreground = fgGMM.predict(test_image) >= bgGMM.predict(test_image)

mask = np.zeros(test_image.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
mask[probably_foreground] = 3
mask[~probably_foreground] = 2

cv2.grabCut(test_image, mask, None, bgGMM.get_params_opencv, fgGMM.get_params_opencv, 3, cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
output = cv2.bitwise_and(test_image, test_image, mask=mask2)

alpha = np.expand_dims(mask2, 2)
outpng = np.append(test_image, alpha, axis=2)
cv2.imwrite("test_foreground.png", outpng)



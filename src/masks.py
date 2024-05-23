import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.exposure import adjust_sigmoid
from scipy import ndimage as nd
from matplotlib import pyplot as plt


def get_binary_tmrm(tmrm, mask_img_path=None):
    denoise_tmrm = denoise_tv_chambolle(tmrm, weight = 0.05)
    denoise_tmrm2 = np.uint8((denoise_tmrm*(255/np.max(denoise_tmrm))).astype(int))
    kernel = np.ones((3, 3), np.uint8)
    denoise_tmrm2 = cv2.erode(denoise_tmrm2, kernel, iterations=3)

    ret_tmrm, thresh_tmrm = cv2.threshold(denoise_tmrm2, np.min(denoise_tmrm2), np.max(denoise_tmrm2), cv2.THRESH_OTSU)
    tmrm_2 = adjust_sigmoid (denoise_tmrm2, (ret_tmrm/np.max(denoise_tmrm2))+0.04, 7)
    tmrm_2 = np.uint8(tmrm_2)
    ret_tmrm2, binary_tmrm = cv2.threshold(tmrm_2, 0, 255, cv2.THRESH_OTSU)
    
    if mask_img_path is not None:
        plt.imsave(mask_img_path,
                   binary_tmrm,
                   cmap="gray")
    return binary_tmrm


def get_binary_img(img):
    background = np.argmax(np.bincount(list(img[img>0].flatten())))         
    ret, thresh = cv2.threshold(img[img>background], 0, 255, cv2.THRESH_OTSU)
    img2 = adjust_sigmoid (img, (ret/255)+0.04, 7)
    img2 = np.uint8(img2)
    ret2, binary2 = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)
    return img2, binary2


def get_binary_nucleus(nucleus, mask_img_path=None):
    denoise_nucleus = nd.median_filter(nucleus, 3)
    kernel = np.ones((3, 3), np.uint8)
    denoise_nucleus2 = cv2.dilate(denoise_nucleus, kernel, iterations=2)

    ret_nucleus, thresh_nucleus = cv2.threshold(denoise_nucleus2, np.min(denoise_nucleus2), np.max(denoise_nucleus2), cv2.THRESH_OTSU)
    nucleus_2 = adjust_sigmoid (denoise_nucleus2, (ret_nucleus/np.max(denoise_nucleus2))+0.04, 7)
    nucleus_2 = np.uint8(nucleus_2)
    ret_nucleus2, binary_nucleus = cv2.threshold(nucleus_2, 0, 255, cv2.THRESH_OTSU)
    binary_nucleus = cv2.dilate(binary_nucleus, kernel, iterations=10)
    binary_nucleus = cv2.erode(binary_nucleus, kernel, iterations=2)
    if mask_img_path is not None:
        plt.imsave(mask_img_path,
                   binary_nucleus,
                   cmap="gray")
    return binary_nucleus

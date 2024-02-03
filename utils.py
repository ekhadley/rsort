import cv2, time, os, platform, math
import numpy as np
import colorgram as cg
from PIL import Image
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

def imscale(img, s):
    assert not 0 in img.shape, "empty src image"
    return cv2.resize(img, (round(len(img[0])*s), round(len(img)*s)), interpolation=cv2.INTER_NEAREST)

def imshow(name, img, s=1.0):
    cv2.imshow(name, imscale(img, s))

def relpix(img, *args):
    h, w, d = img.shape
    if len(args) == 1 and isinstance(args[0], tuple(float)):
        x, y = args[0]
    elif len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], float):
        x, y = args
    else: raise TypeError(f"expected 2 float args or tuple of 2 floats, got args: {args}")
    return int(y*h), int(x*w)

def rotate_image(image, degrees, center=None):
    if center is None:
        h, w, d = image.shape
        center = (w//2, h//2)
    mat = cv2.getRotationMatrix2D(center, angle=degrees, scale=1.0)
    result = cv2.warpAffine(image, mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
lime = '\033[92m'
yellow = '\033[93m'
red = "\033[38;5;196m"
pink = "\033[38;5;206m"
orange = "\033[38;5;202m"
green = "\033[38;5;34m"
gray = "\033[38;5;8m"

bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

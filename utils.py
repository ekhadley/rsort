import cv2, time, os, platform, math
import numpy as np
import colorgram as cg
from PIL import Image
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import scipy

#gry = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#ret, binary = cv2.threshold(gry, 160, 255, cv2.THRESH_BINARY_INV)
#center = [round(np.average(indices)) for indices in np.where(binary >= 255)] 
#cv2.circle(marked, center, 50, (0, 250, 0), 10)

#ret, binary = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY_INV)

def mark_ends(im: np.ndarray, ends: tuple):
    assert len(ends) == 2, f"expected 2 end points, got {len(ends)}"
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2
    out = cv2.circle(im.copy(), (x1, y1), 30, (0, 0, 255), 10)
    out = cv2.circle(out,  (x2, y2), 30, (0, 0, 255), 10)
    return out

def crop_to_ends(im: np.ndarray, points: tuple):
    end1, end2 = points
    x1, y1, x2, y2 = *end1, *end2
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return im.copy()[y1-50:y2+50, x1-50:x2+50]

def isolate(im: np.ndarray, ends: tuple):
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2
    a = math.degrees(math.atan2(y2-y1, x2-x1))
    a = abs(a)
    #if a < 0: a += 360
    out = rotate_image(im, -a, center=(x1, y1))
    dist = int(np.linalg.norm((x1-x2, y1-y2)))
    #out = out[y1-30:y1+30, x1+50:x1+dist-50].astype(np.float32)
    out = out[y1-30:y1+30, x1+50:x1+dist-50]
    return out # the slice used for color gradient examination

def visualize_bands(colors, scale=5.0):
    s = int(100*scale)
    out = np.zeros((s, s*len(colors), 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        col = tuple([int(c) for c in color])
        out = cv2.rectangle(out, (s*i, 0), (s*(i+1), s), color=col, thickness=-1)
    return out

def mark_bands(strp, bandpos):
    out = np.array(strp, copy=True)
    for band in bandpos:
        out = cv2.rectangle(out, (band-2, 0), (band+2, strp.shape[0]), (100, 10, 250), -1)
    return out


def load_test_im(name):
    system = platform.system()
    if system == "Windows": idir = "D:\\wgmn\\rsort\\ims"
    elif system == "Linux": idir = "/home/ek/Desktop/wgmn/rsort/ims"
    else: raise FileNotFoundError(f"{bold+red}unknown system: {system}. failed to load image{endc}")
    path = os.path.join(idir, name)
    return cv2.imread(path)

def imscale(img, s):
    assert not 0 in img.shape, "empty src image"
    return cv2.resize(img, (round(len(img[0])*s), round(len(img)*s)), interpolation=cv2.INTER_NEAREST)

def imshow(name, img, s=1.0, wait=False):
    cv2.imshow(name, imscale(img, s))
    if wait: cv2.waitKey(0)

def _relpix(img, *args):
    h, w, _ = img.shape
    if len(args) == 1 and isinstance(args[0], tuple):
        x, y = args[0]
    elif len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], float):
        x, y = args
    else: raise TypeError(f"expected 2 float args or tuple of 2 floats, got args: {args}")
    return int(y*h), int(x*w)

def rotate_image(image, degrees, center=None):
    if center is None:
        h, w, _ = image.shape
        center = (w//2, h//2)
    mat = cv2.getRotationMatrix2D(center, angle=degrees, scale=1.0)
    result = cv2.warpAffine(image, mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def row_cluster(strp, K=1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(np.float32(strp), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((strp.shape)).astype(strp.dtype)

def col_cluster(strp_: np.ndarray, K=15):
    strp = strp_.swapaxes(0, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(np.float32(strp), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(strp.shape).swapaxes(0, 1)

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

import cv2, time, os, platform, math, json, random
import numpy as np
import colorgram as cg
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import scipy

#gry = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#ret, binary = cv2.threshold(gry, 160, 255, cv2.THRESH_BINARY_INV)
#center = [round(np.average(indices)) for indices in np.where(binary >= 255)] 
#cv2.circle(marked, center, 50, (0, 250, 0), 10)

#ret, binary = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY_INV)

def showextras(im, extras):
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=["blue", "green", "red", "black", "orange"])
    cropped, strp, ends, intensity, avgs, bandpos, bandcolors = extras
    ax.plot(avgs)
    ax.plot(intensity)
    ax.plot(bandpos, intensity[bandpos], "o", ms=10, color="orange")
    #ax.plot(10*np.diff(intensity), color="purple")
    imshow('cropped', cropped)
    imshow('marked', mark_ends(im, ends), s=0.25)
    imshow('processed', mark_bands(cropped, bandpos))
    imshow('bin', strp)
    imshow('vis', visualize_bands(bandcolors))
    plt.show()
    cv2.destroyAllWindows()

def lightness(arr):
    assert 3 in arr.shape, f"{yellow}array must contain a dimension of length 3 for RGB values. got input shape: {arr.shape}{endc}"
    cdim = arr.shape.index(3)
    light = np.amax(arr, axis=cdim) - np.amin(arr, axis=cdim)
    return light

def norm(arr, axis=0):
    return np.sqrt(np.sum(np.square(arr), axis=axis))

def mark_ends(im: np.ndarray, ends):
    assert len(ends) == 2, f"expected 2 end points, got {len(ends)}"
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2
    out = cv2.circle(im.copy(), (x1, y1), 30, (0, 0, 255), 10)
    out = cv2.circle(out, (x2, y2), 30, (0, 0, 255), 10)
    return out

def mark_end(im: np.ndarray, end):
    return cv2.circle(im.copy(), end, 30, (0, 0, 255), 10)

def isolate(im: np.ndarray, ends):
    end1, end2 = ends
    if end1[0] > end2[0]: end1, end2 = end2, end1
    x1, y1, x2, y2 = *end1, *end2
    a = math.degrees(math.atan2(y1-y2, x2-x1))
    if a < 0: a += 360
    out = rotate_image(im, -a, center=(x1, y1))
    dist = int(math.sqrt((x1-x2)**2 +  (y1-y2)**2))
    out = out[y1-30:y1+30, x1+70:x1+dist-70]
    return out # the slice used for color gradient examination

def rotate_image(image, degrees, center=None):
    if center is None:
        h, w, _ = image.shape
        center = (w//2, h//2)
    else: center = (int(center[0]), int(center[1]))
    mat = cv2.getRotationMatrix2D(center, angle=degrees, scale=1.0)
    result = cv2.warpAffine(image, mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def visualize_bands(colors, scale=1.0):
    s = int(100*scale)
    out = np.zeros((s, s*len(colors), 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        col = [int(c) for c in color]
        out = cv2.rectangle(out, (s*i, 0), (s*(i+1), s), color=col, thickness=-1)
    return out

def mark_bands(strp, bandpos):
    out = np.array(strp, copy=True)
    for band in bandpos:
        out = cv2.rectangle(out, (band-2, 0), (band+2, strp.shape[0]), (100, 10, 250), -1)
    return out

def get_test_dir():
    system = platform.system()
    if system == "Windows": return "D:\\wgmn\\rsort\\ims"
    elif system == "Linux": return "/home/ek/Desktop/wgmn/rsort/ims"
    else: raise FileNotFoundError(f"{bold+red}unknown system: {system}. failed to load image{endc}")

def load_test_im(name):
    tdir = get_test_dir()
    path = os.path.join(tdir, name)
    return cv2.imread(path)

def save_test_labels(labels):
    tdir = get_test_dir()
    path = os.path.join(tdir, "labels.json")
    with open(path, "w") as f:
        json.dump(labels, f)
    return

def load_test_labels():
    tdir = get_test_dir()
    path = os.path.join(tdir, "labels.json")
    with open(path, "r") as f:
        #labels = json.load(f, object_hook=json_list_parse)
        labels = json.load(f)
    return parse_label_numerics(labels)

def parse_label_numerics(label):
    for key in label.keys():
        val = label[key]
        if isinstance(val, dict):
            label[key] = parse_label_numerics(val)
        if isinstance(val, list):
            try: label[key] = np.float32(val)
            except ValueError: pass
    return label

def save_test_label(data):
    tdir = get_test_dir()
    path = os.path.join(tdir, "labels.json")
    with open(path, "r") as f:
        labels = json.load(f)
    labels[data["name"]] = data
    with open(path, "w") as f:
        json.dump(labels, f)
    return

def imscale(img, s):
    assert not 0 in img.shape, "empty src image"
    if len(img.shape) == 2: h, w = img.shape
    else: h, w, _ = img.shape
    return cv2.resize(img, (round(w*s), round(h*s)), interpolation=cv2.INTER_NEAREST)

def imshow(name, img, s=1.0, wait=False):
    cv2.imshow(name, imscale(img, s))
    if wait: cv2.waitKey(0)

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

def resistor_value(_colors, reverse=0):
    colors = list(reversed(_colors)) if reverse else _colors
    val = 0
    for i in range(len(colors)-2):
        bandval = color_code[colors[i]]
        val = val*10 + bandval
    val *= 10**color_code[colors[-2]]
    return val

color_code = {"black": 0,
              "brown": 1,
              "red": 2,
              "orange": 3,
              "yellow": 4,
              "green": 5,
              "blue": 6,
              "purple": 7,
              "gray": 8,
              "white": 9,
              "gold": -1,
              "silver": -2}
tolerance_color_code = {"gold": 0.05, "silver": 0.1}

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
allcolors = [purple, blue, cyan, lime, yellow, red, pink, orange, green, gray]



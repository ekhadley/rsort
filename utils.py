import cv2, time, os, platform, math, json, random
import numpy as np
#import colorgram as cg
import tqdm
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import scipy

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
reverse_color_code = {v:k for k, v in color_code.items()}
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
    #ax.plot(7*np.diff(intensity), color="purple")
    imshow('marked', mark_ends(im, ends), s=0.25)
    imshow('processed', mark_bands(cropped, bandpos), s=2.0)
    #imshow('bin', strp)
    imshow('vis', visualize_bands(bandcolors), s=2.0)
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
        out = cv2.rectangle(out, (band-1, 0), (band+1, strp.shape[0]), (200, 200, 250), -1)
    return out

def get_test_dir(tdname = "ims2"):
    system = platform.system()
    if system == "Windows": return f"D:\\wgmn\\rsort\\{tdname}"
    elif system == "Linux": return f"/home/ek/Desktop/wgmn/rsort/{tdname}"
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
    if "none" in colors and colors[-1] != "none": assert False, f"{red}missing color label in non-tolerance position{endc}"
    val = 0
    for color in colors[:-2 if len(colors) > 3 else 2]:
        bandval = color_code[color]
        val = val*10 + bandval
    if len(colors) >= 4: val *= 10**color_code[colors[-2]]
    else: val *= 10**color_code[colors[-1]]
    return val

def color_data_from_labels(labels, keepnone=False, t=None):
    data = {}
    clabel = {k:[] for k in data.keys()}
    for name in labels:
        label = labels[name]
        for i, color in enumerate(label['colors']):
            clabel = label['labels'][i]
            if keepnone or clabel != "none":
                if clabel in data.keys():
                    data[clabel].append(color)
                else:
                    data[clabel] = [color]
    return {k:np.array(v) for k, v in data.items()}

def visualize_color_clusters(labels, colorspace='hsl', t=None, keepnone=False):
    space = colorspace.lower()

    obs = color_data_from_labels(labels, keepnone=keepnone)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.set_xlim((0, 255));ax.set_ylim((0, 255));ax.set_zlim((0, 255))
    if space == 'rgb': labelaxes(ax, 'red', 'green', 'blue')
    elif space =='hls': labelaxes(ax, 'hue', 'light', 'saturation')
    elif space =='ycrcb': labelaxes(ax, 'luma', 'red diff', 'blue diff')
    elif space =='yuv': labelaxes(ax, 'luma', 'U', 'V')
    else: assert False, f"unrecognized colorspace: '{colorspace}'"
    keys = list(obs.keys())
    if keepnone: keys.append("none")
    for col in keys:
        if col in obs.keys(): cols = obs[col]
        else: continue
        if t is not None: cols = cols.copy() @ t
        if space == 'hls': cols = cv2.cvtColor(np.array([cols]), cv2.COLOR_BGR2HLS)[0]
        if space == 'ycrcb': cols = cv2.cvtColor(np.array([cols]), cv2.COLOR_BGR2YCrCb)[0]
        if space == 'yuv': cols = cv2.cvtColor(np.array([cols]), cv2.COLOR_BGR2YUV)[0]
        ax.scatter(cols[:,0], cols[:,1], cols[:,2], color=col if col!='none' else 'pink', s=10)

    #plt.show()

def labelaxes(ax, *args):
    assert 1 < len(args) < 4, f'2 or 3 labels are required. got {len(args)}: {args}'
    ax.set_xlabel(args[0])
    ax.set_ylabel(args[1])
    if len(args) == 3: ax.set_zlabel(args[2])

def grade_metric(data, metric, t=None):
    allscores = []
    colorscores = {k:[] for k in data.keys()}
    for clabel, cvalues in data.items():
        for cval in cvalues:
            out = metric(cval, data, t=t)
            if isinstance(out, tuple): label, *_ = out
            else: label = out
            allscores.append(label == clabel)
            colorscores[clabel].append(label == clabel)
    avgallscores = np.mean(allscores)
    avgcolorscores = {k: round(np.mean(v), 3) for k, v in colorscores.items()}
    coloravg = np.mean(list(avgcolorscores.values()))
    return avgallscores, coloravg, avgcolorscores

def grade_identification(identify, inspect=False):
    tdir, labels = get_test_dir(), load_test_labels()
    imnames = [e for e in os.listdir(tdir) if e.endswith(("png", "jpg", "jpeg"))]
    score = []
    for i in (t:=trange(len(imnames), ncols=100)):
        imname = imnames[i]
        im = load_test_im(imname)
        label = labels[os.path.join(tdir, imname)]
        try:
            info, *extras = identify(im)
        except Exception as e:
            print(f"{bold+red}\nfailed to identify: {imname} with exceptione {e}{endc}")
            continue

        match = label['value'] == info['value']
        if inspect and not match:
            print_data(info)
            print(f"{bold+purple}true value: {label['value']}\n")
            showextras(im, extras)
        score.append(match)

        t.set_description(f"{blue}score: {np.mean(score):.4f}")
    return np.mean(score)

def compute_lookup_array(data, t):
    look = np.zeros((255, 255, 255), dtype=np.int8)
    data = {k:v@t for k, v in data.items()}
    for r in trange(255):
        for g in range(255):
            for b in range(255):
                val = np.array([r, g, b])
                labelidx, mindist = 0, 1e9
                for i, cvalues in enumerate(data.values()):
                    d = np.mean(np.linalg.norm(val@t - cvalues, axis=1, ord=2))
                    if d < mindist:
                        labelidx = i
                        mindist = d
                look[r,g,b] = labelidx
    return look

def print_data(data):
    print(f"{yellow}name: {data['name']}")
    print(f"{orange}ends: {data['ends']}")
    print(f"{pink}bands: {data['bands']}")
    print(f"{green}colors: {data['colors']}")
    print(f"{blue}labels: {data['labels']}{endc}")
    print(f"{lime}reversed: {data['reversed']}{endc}")
    print(f"{cyan}value: {data['value']}{endc}")
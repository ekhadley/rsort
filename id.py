from utils import *


system = platform.system()
if system == "Windows":
    idir = "D:\\wgmn\\rsort\\ims"
elif system == "Linux":
    idir = "//home//ek//Desktop//wgmn//rsort//ims"

def crop_to_ends(im: np.ndarray, points: tuple[float]):
    end1, end2 = points
    x1, y1, x2, y2 = *end1, *end2
    px1, py1, px2, py2 = *relpix(im, x1, y1), *relpix(im, x2, y2)
    if px1 > px2: px1, px2 = px2, px1
    if py1 > py2: py1, py2 = py2, py1
    return im.copy()[py1-50:py2+50, px1-50:px2+50]

def mark_ends(im: np.ndarray, ends: tuple[float]):
    assert len(ends) == 2, f"expected 2 end points, got {len(ends)}"
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2

    out = cv2.circle(im.copy(), relpix(im, x1, y1), 30, (0, 0, 255), 10)
    out = cv2.circle(out, relpix(im, x2, y2), 30, (0, 0, 255), 10)
    return out

def isolate(im: np.ndarray, ends: tuple[float], crop=True):
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2
    px1, py1, px2, py2 = *relpix(im, x1, y1), *relpix(im, x2, y2)
    a = math.degrees(math.atan2(py2-py1, px2-px1))
    a = abs(a)
    #if a < 0: a += 360
    out = rotate_image(im, -a, center=(px1, py1))
    if crop:
        dist = int(np.linalg.norm((px1-px2, py1-py2)))
        out = out[py1-200:py1+200, px1+50:px1+dist-50]
        #out = out[py1-20:py1+20, px1:px1+dist] # the slice used for color gradient examination
    return out

def gradient(im: np.ndarray, retim=True):
    h, w, d = im.shape
    my = h//2
    strp = im[my-30:my+30,:].astype(np.float32)
    strp = row_cluster(strp, K=1)
    #strp = col_cluster(strp, K=15)
    #strp -= np.array([116.06326531, 90.58605442, 36.85034014])
    #strp -= np.mean(strp, axis=(0,1))
    
    avgs = np.mean(strp, axis=(0,)) # average along columns of rgb values
    sums = np.sum(strp, axis=(0,)) # sum along columns of rgb values
    mag = np.mean(strp, axis=(0,2)) # pixel 'magnitude' averaged vertically and horizontally
    #relmags = avgs/mag.reshape(-1, 1)
    relmags = sums/np.sum(strp, axis=(0,2)).reshape(-1, 1)
    return strp, relmags

def row_cluster(strp, K=1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(np.float32(strp), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((strp.shape))

def col_cluster(strp_: np.ndarray, K=15):
    strp = strp_.swapaxes(0, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(np.float32(strp), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
   
    print(f"{red+bold}strp has shape {strp.shape}{endc}")
    print(bold, yellow, label.shape, endc)
    print(bold, gray, center, endc)
    print(bold, gray, center.shape, endc)
    print(bold, pink, res.shape, endc)
    return res.reshape(strp.shape).swapaxes(0, 1)


def endpoints(im: np.ndarray):
    #return (.48, .48), (.305, .76) # 0
    return (.41, .4), (.25, .665) # 1

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["blue", "green", "red"])
qwe, asd = plt.subplots()
asd.set_prop_cycle(color=["blue", "green", "red"])

name = "1.png"
path = os.path.join(idir, name)
if __name__ == "__main__":
    im = cv2.imread(path)
    #im = cv2.GaussianBlur(im, (13,13), 0)
    im = cv2.bilateralFilter(im, 15, 75, 75) 
    
    print(f"{yellow}loaded image at {path}{endc}")
    print(f"{green}image has dimensions:{im.shape}{endc}")

    ends = endpoints(im)
    marked = mark_ends(im, ends)
    rotated = isolate(im, ends)

    strp, colors = gradient(rotated)
    print(colors.shape)
    #extr = cg.extract(Image.fromarray(np.flip(strp, axis=-1)), 6)
    #for c in extr:
    #    print(bold, red, c, endc)

    ax.plot(colors)

    dcolors = np.diff(colors, axis=0)
    print(bold, pink, dcolors.shape, endc)
    asd.plot(dcolors)
    
    #imshow('im', marked, .25)
    #imshow('rotated', rotated, .25)
    #imshow('sliced', strp)
    plt.show()
    cv2.waitKey(0)

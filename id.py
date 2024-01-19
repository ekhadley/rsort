from utils import *

system = platform.system()
if system == "Windows":
    idir = "D:\\wgmn\\rsort\\ims"
elif system == "Linux":
    idir = "//home//ek//Desktop//wgmn//rsort//ims"

name = "0.png"
path = os.path.join(idir, name)

def endpoints(im: np.ndarray):
    return (.48, .48), (.305, .76)

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

def normalize(im: np.ndarray, ends: tuple[float], crop=True):
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2
    px1, py1, px2, py2 = *relpix(im, x1, y1), *relpix(im, x2, y2)
    a = math.degrees(math.atan2(py2-py1, px2-px1))
    a = abs(a)
    #if a < 0: a += 360
    out = rotate_image(im, -a, center=(px1, py1))
    if crop:
        dist = int(np.linalg.norm((px1-px2, py1-py2)))
        out = out[py1-200:py1+200, px1:px1+dist]
        #out = out[py1-20:py1+20, px1:px1+dist] # the slice used for color gradient examination
    return out

def gradient(im: np.ndarray, retim=True):
    h, w, d = im.shape
    my = h//2
    s = im[my-30:my+30,:]
    g = np.median(s, axis=0)
    if retim: return g, s
    return g

import matplotlib .pyplot as plt

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["blue", "green", "red"])
if __name__ == "__main__":
    im = cv2.imread(path)
    im = cv2.GaussianBlur(im, (13,13), 0)
    
    print(f"{yellow}loaded image at {path}{endc}")
    print(f"{lime}image has type:{type(im)}{endc}")
    print(f"{green}image has dimensions:{im.shape}{endc}")

    ends = endpoints(im)
    marked = mark_ends(im, ends)
    rotated = normalize(im, ends, crop=True)
    grad, slic = gradient(rotated)

    ax.plot(grad)

    imshow('im', marked, .25)
    imshow('rotated', rotated, .25)
    imshow('sliced', slic)
    plt.show()
    cv2.waitKey(0)

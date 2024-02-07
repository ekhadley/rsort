from utils import *

system = platform.system()
if system == "Windows":
    idir = "D:\\wgmn\\rsort\\ims"
elif system == "Linux":
    idir = "/home/ek/Desktop/wgmn/rsort/ims"


def endpoints(im: np.ndarray):
    #return (.48, .48), (.31, .75) # im0
    #return (.41, .4), (.25, .67) # im1
    return (.34, 0.985), (.15, .758) # im2

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
    dist = int(np.linalg.norm((px1-px2, py1-py2)))
    out = out[py1-30:py1+30, px1+50:px1+dist-50].astype(np.float32)
    return out # the slice used for color gradient examination

def gradient(strp_: np.ndarray):
    strp = strp_.copy()
    #strp = cv2.cvtColor(strp, cv2.COLOR_BGR2HV)
    strp = row_cluster(strp, K=1)
    #strp = col_cluster(strp, K=25) # cluster the rows 
    #strp -= np.array([116.06326531, 90.58605442, 36.85034014]) # subtract the bluey base resistor color
    #strp -= np.mean(strp, axis=(0,1))
    
    avgs = np.mean(strp, axis=(0,)) # average color along columns of rgb values
    sums = np.sum(strp, axis=(0,)) # sum of rgbs along columns
    mag = np.mean(strp, axis=(0,2)) # average color value (average of averages of rgb) along columns
    ints = np.mean(np.sqrt(np.sum(np.square(strp), axis=2)), axis=(0)) # average 'intensities' of each column ( intensity = sqrt(R**2 + G**2 + B**2))
    
    relmags = avgs/ints.reshape(-1, 1) # plots the relative magnitude of each color compared to total magnitude
    
    #Bavg = np.mean(strp[:,0,:])
    #Gavg = np.mean(strp[:,1,:])
    #Ravg = np.mean(strp[:,2,:])
    #prelmags = np.mean(strp, axis=(0))
    #prelmags[:,0] /= Bavg
    #prelmags[:,1] /= Gavg
    #prelmags[:,2] /= Ravg
    return strp, ints

def row_cluster(strp, K=1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(np.float32(strp), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((strp.shape)).astype(strp.dtype)

def col_cluster(strp_: np.ndarray, K=15):
    strp = strp_.swapaxes(0, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(np.float32(strp), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(strp.shape).swapaxes(0, 1)

name = "2.png"
path = os.path.join(idir, name)

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["blue", "green", "red"])
if __name__ == "__main__":
    im = cv2.imread(path)
    #im = cv2.GaussianBlur(im, (13,13), 0)
    im = cv2.bilateralFilter(im, 25, 75, 75) 
    
    print(f"{yellow}loaded image at {path}{endc}")
    print(f"{green}image has dimensions:{im.shape}{endc}")

    ends = endpoints(im)
    marked = mark_ends(im, ends)
    cropped = isolate(im, ends, crop=True)
    strp, colors = gradient(cropped)

    #extr = cg.extract(Image.fromarray(np.flip(strp, axis=-1)), 6) # color picking for groups

    ax.plot(colors)
    
    imshow('im', marked, .25)
    imshow('cropped', cropped)
    imshow('processed', strp)
    plt.show()
    cv2.waitKey(0)

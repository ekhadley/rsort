from utils import *

def endpoints(im: np.ndarray):
    #return (.475, .489), (.315, .747) # im0
    return (.41, .4), (.25, .67) # im1
    #return (.34, 0.985), (.15, .758) # im2

def crop_to_ends(im: np.ndarray, points: tuple):
    end1, end2 = points
    x1, y1, x2, y2 = *end1, *end2
    px1, py1, px2, py2 = *relpix(im, x1, y1), *relpix(im, x2, y2)
    if px1 > px2: px1, px2 = px2, px1
    if py1 > py2: py1, py2 = py2, py1
    return im.copy()[py1-50:py2+50, px1-50:px2+50]

def mark_ends(im: np.ndarray, ends: tuple):
    assert len(ends) == 2, f"expected 2 end points, got {len(ends)}"
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2

    out = cv2.circle(im.copy(), relpix(im, x1, y1), 30, (0, 0, 255), 10)
    out = cv2.circle(out, relpix(im, x2, y2), 30, (0, 0, 255), 10)
    return out

def isolate(im: np.ndarray, ends: tuple):
    end1, end2 = ends
    x1, y1, x2, y2 = *end1, *end2
    px1, py1, px2, py2 = *relpix(im, x1, y1), *relpix(im, x2, y2)
    a = math.degrees(math.atan2(py2-py1, px2-px1))
    a = abs(a)
    #if a < 0: a += 360
    out = rotate_image(im, -a, center=(px1, py1))
    dist = int(np.linalg.norm((px1-px2, py1-py2)))
    #out = out[py1-30:py1+30, px1+50:px1+dist-50].astype(np.float32)
    out = out[py1-30:py1+30, px1+50:px1+dist-50]
    return out # the slice used for color gradient examination

def gradient(strp_: np.ndarray):
    strp = strp_.copy().astype(np.float32)
    base = np.array([134.06326531, 99, 45]) # subtract the bluey base resistor color
    strp -= base
    #strp -= np.mean(strp, axis=(0,1))
    
    avgs = np.mean(strp_, axis=(0,)) # average color along columns of rgb values
    #sums = np.sum(strp, axis=(0,)) # sum of rgbs along columns
    #mag = np.mean(strp, axis=(0,2)) # average color value (average of averages of rgb) along columns
    ints = np.mean(np.sqrt(np.sum(np.square(strp), axis=2)), axis=(0))

    bands, _ = scipy.signal.find_peaks(ints,
                                    height=2,
                                    threshold=None,
                                    distance=50,
                                    prominence=15,
                                    wlen=None,
                                    width=None,
                                    rel_height=0.5,
                                    plateau_size=None)
    colors = [np.mean(avgs[band-10:band+10], axis=(0)) for band in bands]
    return strp.astype(np.uint8), avgs, ints, bands, colors

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

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["blue", "green", "red", "black"])
if __name__ == "__main__":
    im = load_test_im( "1.png")
    #print(f"{green}image has dimensions:{im.shape}{endc}")
    #im = cv2.GaussianBlur(im, (13,13), 0)
    im = cv2.bilateralFilter(im, 25, 75, 75) 

    ends = endpoints(im)
    marked = mark_ends(im, ends)
    cropped = isolate(im, ends)
    strp, avgs, intensity, bands, colors = gradient(cropped)
    
    #[print(f"{blue}{c.rgb}: {c.proportion:.3f}{endc}") for c in cg.extract(Image.fromarray(np.flip(strp, axis=-1)), 5)]
    
    ax.plot(avgs)
    ax.plot(intensity)
    ax.plot(bands, intensity[bands], "o", ms=10, color="orange" )
    
    for band in bands:
        cropped = cv2.rectangle(cropped, (band-2, 0), (band+2, len(cropped)), color=(50,0,250), thickness=-1)

    ret, binary = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY_INV)
    center = [ round(np.average(indices)) for indices in np.where(binary >= 255) ] 
    cv2.circle(marked, center, 50, (0, 250, 0), 10)
    
    imshow('im', marked, .25)
    imshow('cropped', cropped)
    imshow('processed', strp)
    #imshow('bin', binary, 0.25)
    imshow('vis', visualize_bands(colors), 0.25)
    plt.show()
    cv2.waitKey(0)

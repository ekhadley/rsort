from utils import *
np.set_printoptions(suppress=True)

def isolate(im: np.ndarray, ends, vclip=30, hclip=60):
    end1, end2 = ends
    if end1[0] > end2[0]: end1, end2 = end2, end1
    x1, y1, x2, y2 = *end1, *end2
    dist = int(math.sqrt((x1-x2)**2 +  (y1-y2)**2))
    a = math.degrees(math.atan2(y1-y2, x2-x1))
    if a < 0: a += 360
    out = np.pad(im, [(0,dist), (0,dist), (0, 0)], mode='constant')
    out = rotate_image(out, -a, center=(x1, y1))
    out = out[y1-vclip:y1+vclip, x1+hclip:x1+dist-hclip]
    return out # the slice used for color gradient examination

def survey_test_dir():
    tdir = get_test_dir()
    labels = load_test_labels()
    imnames = [e for e in os.listdir(tdir) if e.endswith(("png", "jpg", "jpeg"))]
    for i, imname in enumerate(imnames):
        print(f"{bold+gray+underline}checking {imname}{endc}")
        im = load_test_im(imname)
        info, *extras = identify(im)
        print_data(info)
        print(f"{bold+underline+pink}true value: {labels[os.path.join(tdir, imname)]['value']}{endc}\n")
        showextras(im, extras)

def band_colors(strp: np.ndarray, numColorClusters=3, peakHeight=2, peakDist=65, peakProminence=20, peakWidth=19, relHeight=0.4, bandSampleWidth=10):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(strp.reshape(-1, 3).astype(np.float32), numColorClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    basecol = centers[np.argmax(counts)]

    base = strp - basecol
    avgs = np.mean(base, axis=0)
    ints = np.mean(np.sqrt(np.sum(np.square(base), axis=2)), axis=0)
    #ints = np.convolve(ints, np.ones(peakWidth), mode='valid')/(peakWidth)
    
    bandpos, info = scipy.signal.find_peaks(ints,
                                    height=peakHeight,
                                    threshold=None,
                                    distance=peakDist,
                                    prominence=peakProminence,
                                    wlen=None,
                                    width=peakWidth,
                                    rel_height=relHeight,
                                    plateau_size=None)

    bandcolors = [np.mean(strp[:,band-bandSampleWidth//2:band+bandSampleWidth//2], axis=(0,1)) for band in bandpos]
    return base, avgs, ints, bandpos, np.rint(bandcolors)

def endpoints(im: np.ndarray, lightThresh=40, lowerMass=5000, upperMass=120000):
    light = lightness(im)
    lightmask = (light>lightThresh).astype(np.uint8)
    
    numlabels, labels, values, centroids = cv2.connectedComponentsWithStats(lightmask)

    rmask = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(numlabels):
        mass = values[i,4]
        if (mass > lowerMass) and (mass < upperMass):
            rmask += (labels==i).astype(np.uint8)

    contours, heirarchy = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.concatenate(contours).squeeze()

    hull = cv2.convexHull(contour).squeeze()

    ellipse = cv2.fitEllipse(hull)
    (xc,yc),(d1,d2),_angle = ellipse
    angle = np.radians(_angle) + math.pi/2
    end1 = np.array([xc+d2/2*np.cos(angle), yc+d2/2*np.sin(angle)], dtype=np.int32)
    end2 = np.array([xc-d2/2*np.cos(angle), yc-d2/2*np.sin(angle)], dtype=np.int32)

    #imm = cv2.drawContours(cv2.cvtColor(rmask*255, cv2.COLOR_GRAY2RGB), contours, -1, (250,0,250), 5)
    #imm = cv2.drawContours(imm, [hull], -1, (0,250,250), 5)
    #zzz = cv2.ellipse(cv2.cvtColor(rmask*255, cv2.COLOR_GRAY2RGB), ellipse, (0, 255, 0), 10)
    #zzz = cv2.ellipse(zzz, ellipse, (0, 255, 0), 10)
    #zzz = cv2.circle(zzz, tuple(end1), 50, (0,0,255), 10)
    #zzz = cv2.circle(zzz, tuple(end2), 50, (0,0,255), 10)
    #imshow('ellipseeee', imm, s=0.25)
    #imshow('ellipse', zzz, s=0.25, wait=True)
    return np.array([end1, end2])

def identify(im, log=False):
    h, w, _ = im.shape
    blur = cv2.bilateralFilter(im, 3, 55, 55)
    ends = endpoints(blur)
    if log: print(f"{gray}located endpoint{endc}")
    cropped = isolate(blur, ends)
    if log: print(f"{gray}isolated body{endc}")
    strp, avgs, intensity, bandpos, bandcolors = band_colors(cropped)
    if log: print(f"{gray}located band positions{endc}")
    colorlabels = [lookup_label(color) for color in bandcolors]
    if log: print(f"{gray}labeled band colors{endc}")
    scaled_bandpos = bandpos/strp.shape[1]
    isreversed = is_reversed(scaled_bandpos, colorlabels)
    value = resistor_value(colorlabels, reverse=isreversed)
    if log: print(f"{gray}calculated value{endc}")

    scaled_ends = ends.astype(np.float32)
    scaled_ends[:,0] /= w
    scaled_ends[:,1] /= h
    info = {"name": "",
            "ends": scaled_ends,
            "bands": scaled_bandpos,
            "colors": bandcolors,
            "labels": colorlabels,
            "reversed": int(isreversed),
            "value": value}
    return info, cropped, strp, ends, intensity, avgs, bandpos, bandcolors

def label_color(val, data, t=None):
    if t is None: data = {k:v for k, v in data.items()}
    else: data = {k:v@t for k, v in data.items()}
    dists = {}
    for clabel, cvalues in data.items():
        if t is None: dists[clabel] = np.linalg.norm(val - cvalues, axis=1, ord=2)
        else: dists[clabel] = np.linalg.norm(val@t - cvalues, axis=1, ord=2)
    
    avgs = {k:np.mean(v) for k, v in dists.items()}
    labelidx = np.argmin(list(avgs.values()))
    return list(avgs.keys())[labelidx], avgs

def is_reversed(bandpos, labels):
    if labels[0] in ('gold', 'silver'): return True
    if labels[-1] in ('gold', 'silver'): return False
    avg = sum(bandpos)/len(bandpos)
    return avg > 0.5

def lookup_label(*args, **kwargs):
    val = args[0]
    #idx = lookup[*np.rint(val).astype(np.uint8)]
    #return reverse_color_code[idx]
    
    v = np.rint(val).astype(np.uint8)
    r, g, b = v[0], v[1], v[2]
    idx = lookup[r, g, b]
    return reverse_color_code[int(idx)]

best = np.load('transform.npy')
lookup = np.load("lookup.npy")
if __name__ == "__main__":
    #im = load_test_im("11.png")
    #info, *extras = identify(im)
    #print_data(info)
    #showextras(im, extras)

    #score = grade_identification(identify, inspect=False)
    #print(f"{bold+purple}{score=:.4f}{endc}")

    #survey_test_dir()

    labels = load_test_labels()
    visualize_color_clusters(labels, colorspace='rgb', t=None)
    #visualize_color_clusters(labels, colorspace='rgb', t=best)
    plt.show()

from utils import *
np.set_printoptions(suppress=True)

def band_colors(strp: np.ndarray, numColorClusters=4, peakHeight=2, peakDist=55, peakProminence=20, peakWidth=20, peakRelHeight=0.5, bandSampleWidth=10):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(strp.reshape(-1, 3).astype(np.float32), numColorClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    basecol = centers[np.argmax(counts)]

    imshow('asdfasdf', visualize_bands(centers))

    base = strp - basecol
    avgs = np.mean(base, axis=0)
    ints = np.mean(np.sqrt(np.sum(np.square(base), axis=2)), axis=0)
    #ints = np.mean(lightness(base), axis=0)

    bandpos, _ = scipy.signal.find_peaks(ints,
                                    height=peakHeight,
                                    threshold=None,
                                    distance=peakDist,
                                    prominence=peakProminence,
                                    wlen=None,
                                    width=peakWidth,
                                    rel_height=peakRelHeight,
                                    plateau_size=None)

    bandcolors = [np.mean(strp[:,band-bandSampleWidth//2:band+bandSampleWidth//2], axis=(0,1)) for band in bandpos]
    return base, avgs, ints, bandpos, np.rint(bandcolors)

def endpoints(im: np.ndarray, lightThresh=40, lowerMass=5000, upperMass=120000):
    light = lightness(im)
    lightmask = (light>lightThresh).astype(np.uint8)
    
    numlabels, labels, values, centroids = cv2.connectedComponentsWithStats(lightmask)

    #imshow('lightmask', (lightmask*255).astype(np.uint8), 0.25)

    rmask = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(numlabels):
        mass = values[i,4]
        if (mass > lowerMass) and (mass < upperMass):
            rmask += (labels==i).astype(np.uint8)

    contours, heirarchy = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.concatenate(contours).squeeze()

    hull = cv2.convexHull(contour).squeeze()
    
    center = np.mean(hull, axis=0, dtype=np.int32)
    dists1 = np.sum(np.square(hull-center), axis=1)
    end1 = hull[np.argmax(dists1)]
    
    dists2 = np.sum(np.square(hull-end1), axis=1)
    end2 = hull[np.argmax(dists2)]

    imm = cv2.drawContours(im.copy(), contours, -1, (250,0,250), 5)
    imm = cv2.drawContours(imm, [hull], -1, (0,250,250), 5)
    #imshow("immmmmmmmmmmmmmm", imm, s=0.25)
    return np.array([end1, end2])

def identify(im):
    h, w, _ = im.shape
    blur = cv2.bilateralFilter(im, 10, 75, 75) 
    ends = endpoints(blur)
    cropped = isolate(im, ends)
    strp, avgs, intensity, bandpos, bandcolors = band_colors(cropped)

    scaled_ends = ends.astype(np.float32)
    scaled_ends[:,0] /= w
    scaled_ends[:,1] /= h
    scaled_bandpos = bandpos/strp.shape[1]
    info = {"name": "",
            "ends": scaled_ends,
            "bands": scaled_bandpos,
            "colors": bandcolors,
            "labels": [],
            "reversed": 0,
            "value": -1}
    return info, cropped, strp, ends, intensity, avgs, bandpos, bandcolors

def grade(auto, label):
    print(f"{bold+cyan}auto:{auto['ends']},\nlabel:{label['ends']}{endc}")
    print(f"{bold+green}auto:{auto['bands']}, label:{label['bands']}{endc}")
    for i, color in enumerate(auto['colors']):
        print(f"{bold+blue}colors {i}: auto:{color}, label:{label['colors'][i]} = {label['labels'][i]}{endc}")
    print()

    aends, lends = auto['ends'], label['ends']
    disps = [aends - lends, [aends[1], aends[0]] - lends]
    enddists = [np.linalg.norm(disps[0], axis=1), np.linalg.norm(disps[1], axis=1)]
    enddist = enddists[np.argmin([sum(e) for e in enddists])]
    endscore = max(enddist)
    print(f"{bold+red} aends: {aends}, lends: {lends}")
    print(f"end score: {endscore}{endc}")
    print()

    abands, lbands = auto['bands'], label['bands']
    lbands = list(reversed(1-lbands))
    stretch = True
    if stretch:
        abands -= abands[0]
        lbands -= lbands[0]
        lbands *= abands[-1]/lbands[-1]
    
    banddists = np.abs(abands - lbands)
    print(f"{bold+purple}auto:{abands}\nlabel:{lbands}")
    print(f" largest diff: {max(banddists)}, avg gap: {np.mean(banddists)} (reversed={label['reversed']}){endc}")
    print()

    acolors, lcolors = auto['colors'], label['colors']
    colordisps = acolors - lcolors
    colordists = np.linalg.norm(colordisps, axis=-1)
    colorscore = max(colordists)
    print(f"{bold+blue} color score: {colorscore}{endc}")
    print()

def survey_test_dir():
    tdir = get_test_dir()
    imnames = [e for e in os.listdir(tdir) if e.endswith(("png", "jpg", "jpeg"))]
    for i, imname in enumerate(imnames):
        print(f"{bold+gray+underline}checking {imname}{endc}")
        im = load_test_im(imname)
        info, *extras = identify(im)
        showextras(im, extras)

def label_color(val, data, t=None):
    data = {k:np.array(v)@t for k, v in data.items()}
    dists = {}
    for clabel, cvalues in data.items():
        dists[clabel] = np.linalg.norm(val@t - cvalues, axis=1, ord=2)
    avgs = {k:np.mean(v) for k, v in dists.items()}
    labelidx = np.argmin(list(avgs.values()))
    return list(avgs.keys())[labelidx], avgs

def grade_metric(data, metric, t=None):
    allscores = []
    colorscores = {k:[] for k in data.keys()}
    for clabel, cvalues in data.items():
        for cval in cvalues:
            label, adists = metric(cval, data, t=t)
            allscores.append(label == clabel)
            colorscores[clabel].append(label == clabel)
    avgallscores = np.mean(allscores)
    avgcolorscores = {k: round(np.mean(v), 3) for k, v in colorscores.items()}
    coloravg = np.mean(list(avgcolorscores.values()))
    return avgallscores, coloravg, avgcolorscores

best = np.array([[-0.91104206, -0.76727, 0.42730725],[ 0.70066949,  0.76806536, -0.81296646],[ 0.35724914, -0.30499501, 0.16356933]])
if __name__ == "__main__":
    #survey_test_dir()
    labels = load_test_labels()
    im = load_test_im("1.png")
    info, *extras = identify(im)
    showextras(im, extras)
    
    #ascore, cscores, acscores = grade_metric(data, metric=label_color, t=best)

    #labels = load_test_labels()
    #visualize_color_clusters(labels, colorspace='rgb')
    #visualize_color_clusters(labels, colorspace='rgb', t=best)
    #plt.show()
"""
parameters:
    light color (200, 190, 30, brightness=0.2)
    blur effect:
        - kernel size
        - strength
        - blur type
    localization:
        - lightness binary threshold
        - connected component area min/max
        - resistor body cropping size/shape
    color identification:
        - color clustering:
            - kmeans criteria
            - number of color clusters
        - intensity peak finding:
            - peak height height
            - peak to peak min distance
            - prominence
        - peak horizontal averaging range
"""

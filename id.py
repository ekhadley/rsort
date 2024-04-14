from utils import *


def band_colors(strp: np.ndarray, numColorClusters=3, peakHeight=2, peakDist=50, peakProminence=20, peakWidth=20, peakRelHeight=0.5, bandSampleWidth=10):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(strp.reshape(-1, 3).astype(np.float32), numColorClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    basecol = centers[np.argmax(counts)]

    base = strp - basecol
    avgs = np.mean(base, axis=0)
    #ints = np.mean(np.sqrt(np.sum(np.square(base), axis=2)), axis=0)
    ints = np.mean(lightness(base), axis=0)

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

def endpoints(im: np.ndarray, lightThresh=55, lowerMass=5000, upperMass=120000):
    light = lightness(im)
    lightmask = (light<lightThresh).astype(np.uint8)
    
    numlabels, labels, values, centroids = cv2.connectedComponentsWithStats(lightmask)

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
    imshow("immm", imm, s=0.25)
    return np.array([end1, end2])

def identify(im):
    h, w, _ = im.shape
    blur = cv2.bilateralFilter(im, 30, 75, 75) 
    ends = endpoints(im)
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
    #lbands = list(reversed(1-lbands) if label['reversed'] else lbands
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

def label_colors(colors):
    c1 = {"black": [71, 62, 53],
          "brown": [70, 64, 77],
          "red":   [81, 65, 115],
          "orange":[],
          "yellow":[66, 124, 103],
          "green": [],
          "blue":  [118, 71, 52],
          "purple":[109, 69, 58],
          "gray":  [96, 88, 100],
          "white": [],
          "gold":  [110, 116, 124],
          "silver":[]}
    labels = []
    s = {}
    for color in colors:
        blab, dist = "", 1e9
        for lab, bgr in c1.items():
            if len(bgr) > 0:
                ndist = np.linalg.norm(np.array(bgr)-np.array(color))
                s[lab] = ndist
                if ndist < dist: blab, dist = lab, ndist
        print(bold, red, color)
        print(orange, s)
        print(green, blab, endc)
        print()
        s = {}
        labels.append(blab)
    return labels

# todo: orientation detection, color labeling
# unless... :flushed:
# we don't **technically** need to know the value of a resistor to sort them.
# we just need to put  them into bins of their own kind.
# orientation detection and color labelling are only actually necessary for 
# determining ohm value, not for differnetiating two resistors
# except aactualyyyyyyyy a resistor can have the same colors in opposite
# order, so we maybe can't just look for the orientation of minimal distance
# cause could give false positive? idk.
# so basically if a resistor's colors match NONE of the previously seen ones,
# we don't have to check orientation to declare it as new. If it has the same
# colors (forward or back) as any other we have to check orientation.
# also we should 
np.set_printoptions(suppress=True)
if __name__ == "__main__":
    #im = cv2.imread("D:\\wgmn\\rsort\\abc\\32.png")
    #im = load_test_im("32.png")
    #info, *extras = identify(im)
    #labels = load_test_labels()ed
    #label = labels["D:\\wgmn\\rsort\\ims5\\32.png"]
    #label = labels["/home/ek/Desktop/wgmn/rsort/ims5/32.png"]

    #grade(info, label)
    #showextras(im, extras)
    #cv2.destroyAllWindows()

    #visualize_color_clusters(labels, colorspace='rgb')
    
    #im = cv2.imread("D:\\wgmn\\rsort\\abc\\32.png")
    im = cv2.imread("D:\\wgmn\\rsort\\abc\\0.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)

    imshow('h', im[:,:,0], s=0.25)
    imshow('l', im[:,:,1], s=0.25)
    imshow('s', im[:,:,2], s=0.25)
    imshow('im', im, s=0.25, wait=True)

    plt.show()
#colorizer
"""
parameters:
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

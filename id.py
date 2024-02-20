from os import walk
from eval import print_data
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

def endpoints(im: np.ndarray, lightThresh=35, lowerMass=5000, upperMass=120000):
    #light = (np.amax(im, axis=-1) - np.amin(im, axis=-1)) # lightness from hsl scale
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
    
    center = np.mean(hull, axis=0, dtype=np.int32)
    dists1 = np.sum(np.square(hull-center), axis=1)
    end1 = hull[np.argmax(dists1)]
    
    dists2 = np.sum(np.square(hull-end1), axis=1)
    end2 = hull[np.argmax(dists2)]
    return np.array([end1, end2])

def label_color(colors):
    pass

def identify(im):
    h, w, _ = im.shape
    blur = cv2.bilateralFilter(im, 15, 75, 75) 
    ends = endpoints(blur)
    cropped = isolate(im, ends)
    strp, avgs, intensity, bandpos, bandcolors = band_colors(cropped)

    scaled_ends = ends.astype(np.float32)
    scaled_ends[0] /= w
    scaled_ends[1] /= h
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
    print(f"{bold+cyan}auto:{auto['ends']}, label:{label['ends']}{endc}")
    print(f"{bold+green}auto:{auto['bands']}, label:{label['bands']}{endc}")
    for i, color in enumerate(auto['colors']):
        print(f"{bold+blue}colors {i}: auto:{color}, label:{label['colors'][i]} = {label['labels'][i]}{endc}")

if __name__ == "__main__":
    im = load_test_im("0.png")
    info, *extras = identify(im)
    labels = load_test_labels()
    label = labels["/home/ek/Desktop/wgmn/rsort/ims/0.png"]
    grade(info, label)
    
    #showextras(im, extras)
    cv2.destroyAllWindows()

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

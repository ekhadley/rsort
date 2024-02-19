from os import walk
from utils import *

def band_colors(strp: np.ndarray):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(strp.reshape(-1, 3).astype(np.float32), 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    basecol = centers[np.argmax(counts)]
    print(bold, orange, centers)
    print(pink, 3*counts/strp.size, endc)
    imshow("backgroundcolor", visualize_bands([basecol]))

    #basecol = np.array([134, 99, 45])
    #basecol = np.array([185, 155, 62])
    base = strp - basecol
    avgs = np.mean(base, axis=0)
    #ints = np.mean(np.sqrt(np.sum(np.square(base), axis=2)), axis=0)
    ints = np.mean(lightness(base), axis=0)
    imshow("lkjnsdvf", lightness(base).astype(np.uint8))

    bandpos, _ = scipy.signal.find_peaks(ints,
                                    height=2,
                                    threshold=None,
                                    distance=50,
                                    prominence=20,
                                    wlen=None,
                                    width=20,
                                    rel_height=0.5,
                                    plateau_size=None)
    bandcolors = [np.mean(strp[:,band-5:band+5], axis=(0,1)) for band in bandpos]
    return base, avgs, ints, bandpos, bandcolors

def endpoints(im: np.ndarray):
    #light = (np.amax(im, axis=-1) - np.amin(im, axis=-1)) # lightness from hsl scale
    light = lightness(im)
    lightmask = (light>35).astype(np.uint8)
    
    numlabels, labels, values, centroids = cv2.connectedComponentsWithStats(lightmask)

    rmask = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(numlabels):
        mass = values[i,4]
        if (mass > 5000) and (mass < 120000):
            rmask += (labels==i).astype(np.uint8)

    contours, heirarchy = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.concatenate(contours).squeeze()

    hull = cv2.convexHull(contour).squeeze()
    
    center = np.mean(hull, axis=0, dtype=np.int32)
    dists1 = np.sum(np.square(hull-center), axis=1)
    end1 = hull[np.argmax(dists1)]
    
    dists2 = np.sum(np.square(hull-end1), axis=1)
    end2 = hull[np.argmax(dists2)]
    return end1, end2

fig, ax = plt.subplots()
#ax.set_prop_cycle(color=["blue", "green", "red", "black"])
if __name__ == "__main__":
    im = load_test_im("0.png")
    im = cv2.bilateralFilter(im, 15, 75, 75) 

    ends = endpoints(im)
    cropped = isolate(im, ends)
    strp, avgs, intensity, bandpos, bandcolors = band_colors(cropped)
    
    #ax.plot(avgs)
    ax.plot(intensity)
    ax.plot(bandpos, intensity[bandpos], "o", ms=10, color="orange")

    #imshow('cropped', cropped)
    imshow('marked', mark_ends(im, ends), s=0.5)
    imshow('processed', mark_bands(cropped, bandpos))
    #imshow('bin', binary)
    imshow('vis', visualize_bands(bandcolors))
    plt.show()
    cv2.destroyAllWindows()

"""
parameters:
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

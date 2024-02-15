from os import walk
from utils import *

def band_colors(strp: np.ndarray):
    base = strp - np.array([134.06326531, 99, 45])
    avgs = np.mean(base, axis=0)
    ints = np.mean(np.sqrt(np.sum(np.square(base), axis=2)), axis=0)

    bandpos, _ = scipy.signal.find_peaks(ints,
                                    height=2,
                                    threshold=None,
                                    distance=50,
                                    prominence=15,
                                    wlen=None,
                                    width=None,
                                    rel_height=0.5,
                                    plateau_size=None)
    bandcolors = [np.mean(avgs[band-10:band+10], axis=(0)) for band in bandpos]
    return base, avgs, ints, bandpos, bandcolors

def endpoints(im: np.ndarray):
    #ends = [950, 1231], [1452, 816] # im0
    #ends = [777, 1-62], [1302, 648] # im1
    #ends = [1914, 881], [1473, 388] # im2

    hsl = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    lightmask = (hsl[:,:,2] > 50).astype(np.uint8)
    #lightmasked = im*np.expand_dims(mask, -1) 
    numlabels, labels, values, centroids = cv2.connectedComponentsWithStats(lightmask)
    #[print(bold, e, endc) for e in  values]

    rmask = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(numlabels):
        mass = values[i,4]
        if (mass > 5000) and (mass < 120000):
            rmask += (labels==i).astype(np.uint8)

    contours, heirarchy = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.concatenate(contours).squeeze()
    hull = cv2.convexHull(contour).squeeze()
    #rect = cv2.minAreaRect(hull)
    #rpoints = cv2.boxPoints(rect).astype(np.int32)
    center = np.mean(hull, axis=0, dtype=np.int32)
    dists1 = norm(hull-center, axis=1)
    end1 = hull[np.argmax(dists1)]

    dists2 = norm(hull-end1, axis=1)
    end2 = hull[np.argmax(dists2)]

    im_ = cv2.circle(im.copy(), center, 30, (255, 0, 150), 10)
    im_ = cv2.circle(im_, end1, 30, (255, 0, 150), 10)
    im_ = cv2.circle(im_, end2, 30, (255, 0, 150), 10)
    im_ =  cv2.drawContours(im_, [hull], 0, (100,0,255), 5)
    #im =  cv2.drawContours(im, [rpoints], 0, (0,255,0), 10)
    imshow('im', im_, 0.25)
    return (end1, end2)

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["blue", "green", "red", "black"])
if __name__ == "__main__":
    im = load_test_im("1.png")
    print("jkbhsdfbhjvsdfvbhj")
    #print(f"{green}image has dimensions:{im.shape}{endc}")
    #im = cv2.GaussianBlur(im, (13,13), 0)
    im = cv2.bilateralFilter(im, 35, 75, 75) 

    ends = endpoints(im)
    cropped = isolate(im, ends)
    strp, avgs, intensity, bandpos, bandcolors = band_colors(cropped)
    
    #[print(f"{blue}{c.rgb}: {c.proportion:.3f}{endc}") for c in cg.extract(Image.fromarray(np.flip(im, axis=-1)), 5)]
    
    ax.plot(avgs)
    ax.plot(intensity)
    ax.plot(bandpos, intensity[bandpos], "o", ms=10, color="orange" )
    
    imshow('cropped', cropped)
    imshow('processed', mark_bands(strp, bandpos))
    #imshow('bin', binary, 0.25)
    #imshow('vis', visualize_bands(colors), 0.25)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

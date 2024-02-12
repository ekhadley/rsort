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
    center = [950, 1231], [1452, 816] # im0
    #center = [777, 1-62], [1302, 648] # im1
    #center = [1914, 881], [1473, 388] # im2

    hsl = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    mask = (hsl[:,:,2] > 50).astype(np.uint8)
    masked = im*np.expand_dims(mask, -1)
    stats = cv2.connectedComponentsWithStats(mask)
    print(bold, blue, len(stats), endc)
    print(bold, blue, len(stats[0]), endc)
    print(bold, green, stats[0][4], endc)
    for i in stats[:,cv2.CC_STAT_AREA]:
        print(bold, i, endc)

    
    return center, hsl, mask, masked

fig, ax = plt.subplots()
ax.set_prop_cycle(color=["blue", "green", "red", "black"])
if __name__ == "__main__":
    im = load_test_im("0.png")
    print("jkbhsdfbhjvsdfvbhj")
    #print(f"{green}image has dimensions:{im.shape}{endc}")
    #im = cv2.GaussianBlur(im, (13,13), 0)
    im = cv2.bilateralFilter(im, 35, 75, 75) 

    ends, hsl, mask, masked = endpoints(im)
    marked = mark_ends(im, ends)
    cropped = isolate(im, ends)
    strp, avgs, intensity, bandpos, bandcolors = band_colors(cropped)
    
    #[print(f"{blue}{c.rgb}: {c.proportion:.3f}{endc}") for c in cg.extract(Image.fromarray(np.flip(im, axis=-1)), 5)]
    
    ax.plot(avgs)
    ax.plot(intensity)
    ax.plot(bandpos, intensity[bandpos], "o", ms=10, color="orange" )
    
    imshow('im', marked, .25)
    #imshow('cropped', cropped)
    imshow('processed', mark_bands(strp, bandpos))
    #imshow('bin', binary, 0.25)
    imshow('mask', mask*255, 0.25)
    imshow('masked', masked, 0.25)
    #imshow('vis', visualize_bands(colors), 0.25)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

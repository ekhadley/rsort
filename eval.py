from utils import *

def print_data(data):
    print(f"{yellow}name: {data['name']}")
    print(f"{orange}ends: {data['ends']}")
    print(f"{pink}bands: {data['bands']}")
    print(f"{green}colors: {data['colors']}")
    print(f"{blue}labels: {data['labels']}{endc}")

def pixels(pt, im):
    h, w, _ = im.shape
    return (round(pt[0]*w), round(pt[1]*h))

def markends(im: np.ndarray, ends: tuple):
    global IS
    if len(ends) == 0: return im
    end1 = pixels(ends[0], im)
    out = cv2.circle(im.copy(), end1, 30, (0, 0, 255), 10)
    for end in ends[1:]:
        out = cv2.circle(out, pixels(end, im), 30, (0, 0, 255), 10)
    return out

def markbands(strp: np.ndarray, bands: tuple):
    if len(bands) == 0: return strp
    h, w, _ = strp.shape
    band1 = round(bands[0]*w)
    out = cv2.rectangle(strp.copy(), (band1-2, 0), (band1+2, h), (100, 10, 250), -1)
    for band in bandpos:
        bandx = round(band*w)
        out = cv2.rectangle(out, (bandx-2, 0), (bandx+2, h), (100, 10, 250), -1)
    return out

def endpoint_select(event, x, y, flags, params):
    global mousex, mousey, endpoints, im, IS
    h, w, _ = im.shape
    if event == cv2.EVENT_LBUTTONUP and (0 <= x <= w) and (0 <= y <= h):
        mousex, mousey = x/(w*IS), y/(h*IS)
        if len(endpoints) < 2:
            endpoints.append((mousex, mousey))
        elif len(endpoints) == 2:
            dists = [(end[0] - mousex)**2 + (end[1] - mousey)**2 for end in endpoints]
            if dists[0] < dists[1]: endpoints[0] = (mousex, mousey)
            else: endpoints[1] = (mousex, mousey)

def band_color(strp: np.ndarray, band: tuple):
    h, w, _ = strp.shape
    bpos = round(band*w)
    color = [round(e) for e in np.mean(strp[:,bpos-5:bpos+5], axis=(0,1))]
    return color

def band_select(event, x, y, flags, params):
    global mousex, mousey, bandpos, strp, SS, colors
    if event == cv2.EVENT_LBUTTONUP:
        h, w, _ = strp.shape
        mousex = x/(w*SS)
        for i, band in enumerate(bandpos):
            if abs(band - mousex) < 0.05:
                del bandpos[i], colors[i]
                return
            if mousex < band:
                bandpos.insert(i, mousex)
                colors.insert(i, band_color(strp, mousex))
                return
        if len(bandpos) == 0 or mousex > bandpos[-1]:
            bandpos.append(mousex)
            colors.append(band_color(strp, mousex))

IS = 0.3
SS = 2.0


mousex, mousey = 0, 0
endpoints, bandpos, colors = [], [], []
strpBound = False
imname = "4.png"
if __name__ == "__main__":
    im = load_test_im(imname)

    imshow("im", im, s=IS)
    cv2.setMouseCallback("im", endpoint_select)

    while 1:
        imshow("im", markends(im, endpoints), s=IS)
        if len(endpoints) == 2:
            if endpoints[0][0] > endpoints[1][0]: endpoints = [endpoints[1], endpoints[0]]
            strp = isolate(im, [pixels(end, im) for end in endpoints])
            imshow("strp", markbands(strp, bandpos), s=SS)
            if not strpBound: cv2.setMouseCallback("strp", band_select); strpBound = True
        if len(colors) > 0:
            imshow("bands", visualize_bands(colors, scale=SS))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): cv2.destroyAllWindows(); exit();
        if key == ord('s') and len(endpoints) == 2 and ( 4 <= len(colors) <= 6):
            data = {"name":imname, "ends": endpoints, "bands": bandpos, "colors": colors, "labels": []}
            while len(data["labels"]) < len(data["colors"]):
                label = input(f"Enter label for color {len(data['labels'])}: ")
                if label in color_code.keys(): data["labels"].append(label)
                else: print(f"Invalid color code: {label}. Try again.")
            print(f"{bold}Data collected:{endc}")
            print_data(data)
            if input("To save, press 's' again. To cancel, press 'q': ") == "s":
                append_test_label(data)
                print("Data saved to data.json")
            
            cv2.destroyAllWindows(); exit();



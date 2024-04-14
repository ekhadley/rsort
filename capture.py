from utils import np, cv2, imshow, imscale, time
from picamera2 import Picamera2

pc2 = Picamera2()
stillConf = pc2.create_still_configuration()
pc2.start(config=stillConf)
time.sleep(2)
save = "//home//ek//Desktop//wgmn//rsort//abc//"
c = 7

while 1:
    im = pc2.capture_array()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    marked = imscale(im, .4)

    ms = np.shape(marked)
    marked = cv2.circle(marked, (ms[1]//2, ms[0]//2), 5, (50, 0, 250), 2)

    imshow('im', marked, 0.8)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(f"{save}{c}.png", im)
        print(f"saved: '{c}.png'")
        c += 1
        time.sleep(1)

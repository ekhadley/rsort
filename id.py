from utils import *

system = platform.system()
if system == "Windows":
    idir = "D:\\wgmn\\rsort\\ims"
elif system == "Linux":
    idir = "//home//ek//Desktop//wgmn//rsort//ims"

name = "0.png"
path = os.path.join(idir, name)

def endpoints():
    return (400, 400), (700, 700)

def markends(im, points):
    assert len(points) == 2, f"expected 2 end points, got {len(points)}"
    p1, p2 = points
    x1, y1 = p1
    x2, y2 = p2

    cv2.circle(im, (x1, y1), 30, (0, 0, 255), 10)
    cv2.circle(im, (x2, y2), 30, (0, 0, 255), 10)
    return im

if __name__ == "__main__":
    print(f"{yellow}loaded image at {path}{endc}")
    
    im = cv2.imread(path)
    print(f"{lime}imge has type:{type(im)}{endc}")
    print(f"{green}image has dimensions:{im.shape}{endc}")
    
    ends = endpoints()
    markends(im, ends)

    imshow('im', im, .25)
    cv2.waitKey(0)

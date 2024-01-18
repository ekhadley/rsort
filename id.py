from utils import *

system = platform.system()
if system == "Windows":
    idir = "D:\\wgmn\\rsort\\ims"
elif system == "Linux":
    idir = "//home//ek//Desktop//wgmn//rsort//testes"

name = "0.png"
path = os.path.join(idir, name)

if __name__ == "__main__":
    im = cv2.imread(os.path.join(path, name))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('im', im)
    cv2.waitKey(0)

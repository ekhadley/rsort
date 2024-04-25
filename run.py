from utils import *
from identify import *
import RPi.GPIO as gpio
from gpiozero import Servo
from picamera2 import Picamera2

class binController:
    def __init__(self, pin, num_bins, num_accessible, home=0.75, inc=0.44):
        self.servo = Servo(pin, initial_value=home, min_pulse_width=0.00038, max_pulse_width=0.0024)
        self.pin = pin
        self.bins_home = home
        self.bins_inc = inc

        self.nbins = num_bins
        self.nacc = num_accessible

        self.binvalues = []
        self.count = 0

    def move(self, a):
        t = max(-1, min(a, 1))
        self.servo.value = t
    def reset(self):
        self.servo.value = self.bins_home
    def move_to_bin(self, n):
        self.move(self.bins_home - n*self.bins_inc)
    def drop(self, val):
        if val in self.binvalues:
            door_up()
            self.move_to_bin(self.binvalues.index(val))
            time.sleep(0.5)
            door_down()
            time.sleep(0.25)
            door_up()

            self.count += 1
            return True
        elif len(self.binvalues) < self.nacc:
            door_up()
            self.move_to_bin(len(self.binvalues))
            time.sleep(0.25)
            door_down()
            time.sleep(0.25)
            door_up()
            
            self.binvalues.append(val)
            self.count += 1
            return True
        else:
            return False
    def candrop(self, val):
        if val in self.binvalues: return True
        elif len(self.binvalues) < self.nacc: return True
        else: return False

door_home_position, door_down_position = -0.55, 0.65
def door_up(home=door_home_position): door.value = home
def door_down(down=door_down_position): door.value = down
def light_on():
    os.system('sudo python3 ~/Desktop/wgmn/rsort/lights.py 1')
def light_off():
    os.system('sudo python3 ~/Desktop/wgmn/rsort/lights.py 0')

state = 'looking'
if __name__ == "__main__":
    pc2 = Picamera2()
    stillConf = pc2.create_still_configuration()
    pc2.start(config=stillConf)

    light_on()

    door = Servo(19, initial_value = door_home_position)
    bins = binController(26, 8, 5)
    time.sleep(2)
    

    while 1:
        im = pc2.capture_array()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imshow('im', im, s=0.2)
        
        wait = cv2.waitKey(1) & 0xFF
        if wait == ord('e'):
            valid = False
            try:
                info, *extras = identify(im, log=True)
                print_data(info)
                #showextras(im, extras)
                valid = valid_info(info)
            except Exception as e:
                print(f"{red+bold}identification failed with exception: {e}. discarding results.{endc}")
            cv2.destroyAllWindows()
            if valid and state != 'full':
                if bins.candrop(info['value']):
                    state = 'pending'
                    print(f"{bold+green+underline}info extracted successfully. press 'd' to sort the resistor.{endc}")
                else:
                    state = 'full'
                    print(f"{bold+yellow+underline}info extracted but all bins are occuppied. cannot sort.{endc}")
            else: print("data was invalid")

        if state == 'pending':
            imshow('processed', mark_bands(extras[0], extras[-2]), s=1.0)
            if wait == ord('d'):
                bins.drop(info['value'])
                print(f"{bold+lime}{info['value']}ohm resistor stored in bin {bins.binvalues.index(info['value'])}{endc}\n\n")
                state = 'looking'
            if wait == ord('r'):
                state = 'looking'
            
        if wait == ord('q'):
            cv2.destroyAllWindows()
            light_off()
            exit()



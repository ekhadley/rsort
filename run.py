from utils import *
#from identify import *
import RPi.GPIO as gpio
from gpiozero import Servo
from picamera2 import Picamera2
import board
import import neopixel as neo

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
            print(f"{val} is stored in bin {self.binvalues.index(val)}")
            door_up()
            self.move_to_bin(self.binvalues.index(val))
            time.sleep(0.5)
            door_down()
            time.sleep(0.25)
            door_up()

            self.count += 1
            return True
        elif len(self.binvalues) < self.nacc:
            print(f"{val} is new. storing in bin {len(self.binvalues)}")

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
            print(f"cant store {val}. bins are full ({self.nacc=}, {len(self.binvalues)=})")
            print(f"{bold+yellow} error: all available bins are occupied. cannot accomodate resistor of value {val}{endc}")
            return False

door_home_position, door_down_position = -0.7, 0.65
light_color = (200, 190, 30)
def door_up(home=door_home_position): door.value = home
def door_down(down=door_down_position): door.value = down

if __name__ == "__main__":
    pc2 = Picamera2()
    stillConf = pc2.create_still_configuration()
    pc2.start(config=stillConf)

    light = neo.NeoPixel(board.D18, 16, brightness=0.2)
    light.fill(light_color)
    
    door = Servo(19, initial_value = door_home_position)
    bins = binController(26, 8, 5)
    time.sleep(2)
    

    while 1:
        im = pc2.capture_array()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imshow('image', im, s=0.25)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

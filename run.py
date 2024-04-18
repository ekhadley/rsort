from utils import *
<<<<<<< HEAD
from rsort.identify import *
import RPi.GPIO as gpio
=======
from identify import *
import RPi.GPEIO as gpio
>>>>>>> 4cd803b (oops)
from gpiozero import Servo

class binController:
    def __init__(self, num_bins, num_accessible, pin, home=0.8, inc=0.44):
        self.servo = Servo(pin, initial_value=home, min_pulse_width=0.00038, max_pulse_width=0.0024)
        self.pin = pin
        self.bins_home = home
        self.bins_inc = inc

        self.nbins = num_bins
        self.nacc = num_accessible

        self.binvalues = []
        self.nocc = 0
        self.count = 0

    def move(self, inc):
        self.servo.value = inc
    def reset(self):
        self.servo.value = self.bins_home
    def move_to(self, n):
        self.servo.value = self.bins_home - n*self.bins_inc
    def drop(self, val):
        if val in self.labels:
            self.move_to(self.labels.index(val))
            self.count += 1
            return True
        elif self.nocc < self.nacc:
            self.labels.append(val)
            self.move_to(self.nocc)
            self.count += 1
            return True
        else:
            print(f"{bold+yellow} error: all available bins are occupied. cannot accomodate resistor of value {val}{endc}")
            return False

door_home_position, door_down_position = -0.7, 0.75
def door_up(home=-door_home_position): door.value = home
def door_down(down=door_down_position): door.value = down

if __name__ == "__main__":
    door = Servo(19, initial_value=door_home_position)
    bins = binController(26, 8, 5, initial_value=-1, inc=0.44)

    while 1:
        bins.move_to(0)
        door_down()
        time.sleep(1)
        door_up()
        time.sleep(1)
        bins.move_to(3)
        door_down()
        time.sleep(1)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  adsTest.py
#  
#  Copyright 2023  <pi@raspberrypi>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
#!/usr/bin/env python
import time
import board
import busio
import serial
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from rpi_lcd import LCD
import time
import board
import busio
import serial
import numpy as np
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from rpi_lcd import LCD
import neopixel 


#configure serial comms
srl = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 38400,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)

#create light strip, golden yellow
strip=neopixel.NeoPixel(board.D18,1,brightness=1.0/20)
color=(255,192,0)
coff=(0,0,0)



#max selection based on experimental values
sel_max=26450

def main(args):
    strip.fill((0,0,0))
    d=0.1
    waittime=0.01
    i=0
    while i<3:
        t=time.time()
        srl.write(b'990')
        print(time.time()-t)
        strip[0]=color
        print(time.time()-t)
        time.sleep(d-waittime)
        srl.write(b'800')
        t=time.time()
        strip[0]=coff
        print(time.time()-t)
        time.sleep(waittime)
        i=i+1
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

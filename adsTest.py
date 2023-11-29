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

lcd=LCD()
#create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

#create ADC object
ads= ADS.ADS1115(i2c)

#create single ended input for tempo and instrument and song selection
selector=AnalogIn(ads,ADS.P0)

#max selection based on experimental values
sel_max=26450

def main(args):
    while 1:
        lcd.text(str(selector.value),1)
        time.sleep(1)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

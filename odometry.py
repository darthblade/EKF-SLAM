import socket
import logging
import serial
import struct
from threading import Thread
import numpy as np


class odometry:

    def __init__(self, UDP_PORT=5555):
        UDP_IP = "0.0.0.0"
        self.x = 0  # forwad and backward -forward +backward
        self.y = 0  # left right tilt -right +left
        self.left = 0
        self.right = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0)
        self.odoCurrent = np.array([0, 0, 0, 0, 0, 0])
        self.odoOld = np.array([0, 0, 0, 0, 0, 0])
        self.logger = logging.getLogger('odo')
        logging.basicConfig(level=logging.INFO)
        self.count = 0

    def start(self):
        self.threadRun = True
        handler = Thread(target=self.handler, args=())
        handler.daemon = True
        self.ser.write("r\n".encode('ascii'))  # reset our position
        # make sure the laser is on
        self.ser.write("laseron\n".encode('ascii'))
        handler.start()
        self.logger.info("Odometry started")
        return self

    def stop(self):
        self.threadRun = False
        self.ser.write("r\n".encode('ascii'))  # reset our position
        self.ser.write("laseroff\n".encode('ascii'))  # turn the laser off
        self.ser.write(("0:0\n").encode('ascii'))  # stop movement
        self.ser.close()
        self.sock.close()
        self.logger.info("stopped")
        return self

    def handler(self):
        while self.threadRun:
            # if incoming bytes are waiting to be read from the serial input
            # buffer
            if (self.ser.inWaiting() > 0):
                # get first two bytes of serial to check if we have eol
                data_str = self.ser.read(2)
                # read more bytes until we have a valid end termination
                while data_str[-2:] != b'\r\n':
                    data_str += self.ser.read(1)
                if len(data_str) is 26:  # check length of data to make sure its for odo
                    self.odoCurrent = ([
                        struct.unpack('f', data_str[:4])[0],  # x
                        struct.unpack('f', data_str[4:8])[0],  # y
                        struct.unpack('f', data_str[8:12])[0],  # o
                        struct.unpack('f', data_str[12:16])[0],  # v
                        struct.unpack('f', data_str[16:20])[0],  # w
                        struct.unpack('I', data_str[20:24])[0]])  # micro sec
                    self.odoOld = self.odoCurrent

                    self.logger.debug(
                        'Received Odometry at: %s', self.odoCurrent[5])
                else:
                    self.logger.warning(
                        'Unhandled data, size(%s): %s', len(data_str), data_str)
            try:
                data, addr = self.sock.recvfrom(123)
                dataList = data.decode('utf-8').split(",")
                # the only data we want is from the accellerometer
                if(len(dataList) == 3):
                    x = float(dataList[0].strip())
                    y = float(dataList[1].strip())
                    if(not(x >= 10.5 or x <= -10.5) and (x >= 0.5 or x <= -0.5)):
                        x = -x * 1.5  # 9.6
                    else:
                        x = 0
                    if(not(y >= 10.5 or y <= -10.5) and (y >= 0.5 or y <= -0.5)):
                        y = -y * 1.5  # 9.6
                    else:
                        y = 0
                    if(y > 0):
                        left = (x + y) / 2
                        right = x
                    else:
                        right = (x + abs(y)) / 2
                        left = x

                    if self.left is not left or self.right is not right:
                        self.left = left
                        self.right = right
                        self.ser.write(
                            (str(int(right)) + ":" + str(int(left)) + "\n").encode('ascii'))
                        self.logger.debug('serial TX: %s', [
                                          int(right), int(left)])
                else:
                    self.logger.warning(
                        "Unhandled data from accellerometer: %s", data)
            except socket.error:
                pass

    def getOdometry(self):
        return (self.odoCurrent)

    def toggleUpdateRate(self):  # this will force a high update rate of the odometry
        self.ser.write("c\n".encode('ascii'))

    def toggleLED(self):
        # blink the led instead of toggle on
        self.ser.write("led\n".encode('ascii'))
        self.ser.write("led\n".encode('ascii'))

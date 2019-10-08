import logging
from odometry import odometry as odoManager
from detector import detector as detect
import numpy as np


class SLAM:

    def __init__(self):  # (1640,1232)
        self.odometry = odoManager()

        self.detector = detect(resolution=(640, 400),
                               framerate=32)  # (1640,1232)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)

        self.distance = open('test/dist.dat', 'ab')
        self.position = open('test/pos.dat', 'ab')

    def start(self):
        self.detector.start()
        self.odometry.start()
        return self

    def stop(self):
        self.detector.stop()
        self.odometry.stop()

    def SLAM(self):
        frame, distance = self.detector.getDistanceArrayWithFrame()
        position = self.odometry.getOdometry()
        return position, frame, distance

    def saveToFile(self, p, frame, distance, count):
        self.odometry.toggleLED()
        np.savetxt(self.distance, np.atleast_2d(distance), fmt='%5s', delimiter=",")
        np.savetxt(self.position, np.array([p]), fmt='%5s', delimiter=",")
        np.save('test/frames/frame' + str(count), frame)
        self.odometry.toggleLED()

    def cleanRescources(self):
        self.distance.close()
        self.lines.close()
        self.position.close()
        self.features.close()

    def getCenterPixel(self):
        return self.detector.getAverageCenterPixel()


__name__ = '__main__'

slamprocess = SLAM()
slamprocess.start()

count = 0
while True:
    try:
        p, f, d = slamprocess.SLAM()
        slamprocess.saveToFile(p, f, d, count)
        count += 1
        print("\x1b[1A\x1b[2Kframe:{0}".format(count))
        
    except (KeyboardInterrupt, SystemExit):
        print("\nQuitting...")
        #slamprocess.cleanRescources()
        slamprocess.stop()
        break

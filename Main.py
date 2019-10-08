import logging
from odometry import odometry as odo
from detector import detector as detect
import numpy as np
import zmq
import pickle
import zlib

class SLAM:

    def __init__(self):  # (1640,1232)
        self.odometry = odo()

        self.detector = detect(resolution=(640, 400), framerate=32)  # (1640,1232)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)

        self.port = "5558"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind("tcp://*:%s" % self.port)
        self.sendError = 0

        self.currentOdometryData = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)  # x,y,theta,v,w,elapsed
        self.previousOdometryData = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)  # x,y,theta,v,w,elapsed

        self.predictedPath = np.zeros((2, 2))
        self.path = np.zeros((2, 2))

        # we start with only the state vector without the features
        self.mu = np.array([[0], [0], [0]])
        self.prevMu = self.mu

        self.covariance = np.array([0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.017], dtype=np.float64).reshape((3, 3))  # starting error covariance

        self.alpha = 30  # mahanobis distance gate threshold
        self.Q = np.array([0.15, 0, 0, 1 * np.pi / 180]).reshape((2, 2)) ** 2  # covariance observations
        self.R = np.array([0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.0000185]).reshape(3, 3)  # variance of movement error use stdev with each measurement var

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

    def sendData(self, object):
        """Compress data and send it."""
        pickledObject = pickle.dumps(object, -1)
        compressed = zlib.compress(pickledObject)
        try:
            return self.socket.send(compressed, zmq.NOBLOCK)
        except zmq.Again:
            print("Could not send data, is the host viewer connected?")
            self.sendError += 1
            if self.sendError > 3:
                raise Exception('Could not reach host viewer.')


__name__ = '__main__'

slamprocess = SLAM()
slamprocess.start()

count = 0
k=np.array=([1,2,3])
print("press space to capture image for calibration\n")
while True:
    try:
        slamprocess.sendData(k)
        
    except (KeyboardInterrupt, SystemExit):
        print("\nQuitting...")
        #slamprocess.cleanRescources()
        slamprocess.stop()
        print("done!")
        break

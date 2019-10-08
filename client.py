import zmq
import zlib
import pickle
import numpy as np
from draw import paint
from threading import Thread

class observer:

    def __init__(self, port="5556"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect("tcp://192.168.137.78:%s" % port)
        with np.load('callibrationData/distanceToCoordinateMatrix.npz') as data:
            self.lookupArray = data['array']
        self.rangeViewMaxDist = 0.7  #m
        self.rangeViewMinDist = 0.1  #m
        self.threadRun = True
        self.dataGetter = Thread(target=self.update, args=())
        self.dataGetter.daemon = True
        

    def update(self):
        while self.threadRun:
            self.getAll()

    def getCompressed(self):
        """Get object to deserialize and decompress."""
        zobj = self.socket.recv(0)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

    def getUncompressed(self):
        """Get object and deserialize"""
        return pickle.loads(self.socket.recv(0))

    def getAll(self):
        self.stepCount = self.getUncompressed()
        self.mu = self.getUncompressed()
        self.cov = self.getUncompressed()
        self.currentOdometryData = self.getUncompressed()
        self.frame = self.getUncompressed()
        #self.packetSpeed = self.getUncompressed()

    def convertFrame(self):
        self.frameMax = np.argmax(self.frame[int(np.floor(self.frame.shape[0] / 2)):, :], axis=0)
        resultingMap = self.lookupArray[self.frameMax.astype(np.integer), np.arange(self.frameMax.size)]
        self.laserview2D = np.delete(resultingMap, np.where(np.logical_or(resultingMap[:, 0] > self.rangeViewMaxDist, resultingMap[:, 0] < self.rangeViewMinDist)), axis=0)

    def advanceStep(self):
        #self.getAll()
        self.convertFrame()
        # self.predictedPath = np.vstack((self.predictedPath, self.mu[:2].T))
        # self.odometryPath = np.vstack((self.odometryPath, self.currentOdometryData[:2]))

obs = observer()
obs.getAll()
obs.dataGetter.start()
v = paint()
v.start("View", viewPort=True, output=False)
while True:
    try:
        v.clear()
        obs.advanceStep()
        #v.drawMap(obs.mu, obs.cov, obs.laserview2D)
        v.drawAll(obs.mu, obs.cov, obs.frame, obs.frameMax[::-1], obs.laserview2D)
        #v.drawPath(obs.odometryPath[:, 0], obs.odometryPath[:, 1], colour='red')
        #v.drawPath(obs.predictedPath[:, 0], obs.predictedPath[:, 1], colour='black')
        #print('\r FPS: ' + str(obs.fps) + '\r'),
        v.draw()
    except (KeyboardInterrupt, SystemExit):
        obs.threadRun = False
        print("Quitting..")
        break
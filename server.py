import numpy as np
from scipy.linalg import block_diag
from odometry import odometry as odo
from detector import detector as detect
import zmq
import pickle
import zlib
import logging
from datetime import datetime


class SLAM:

    def __init__(self):

        self.dataPort = "5556"
        self.dataContext = zmq.Context()
        self.dataSocket = self.dataContext.socket(zmq.PUSH)
        self.dataSocket.bind("tcp://*:%s" % self.dataPort)
        self.sendError = 0

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.odo = odo()

        self.sensor = detect(resolution=(640, 400), framerate=32)  # (1640,1232)

        self.stepCount = 0

        self.currentOdometryData = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)  # x,y,theta,v,w,elapsed
        self.previousOdometryData = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)  # x,y,theta,v,w,elapsed

        self.mu = np.array([[0], [0], [0]])  # set this to any starting position
        self.prevMu = self.mu

        self.covariance = np.array([0.01, 0, 0, 0, 0.01, 0, 0, 0, 4.5 * np.pi / 180], dtype=np.float64).reshape((3, 3))  # starting error covariance

        self.alpha = 20  # mahalanobis distance threshold
        self.Q = np.array([0.03, 0, 0, 0.2 * np.pi / 180]).reshape((2, 2)) ** 2  # covariance observations
        self.R = np.array([0.01, 0, 0, 0, 0.01, 0, 0, 0, 4.5 * np.pi / 180]).reshape(3, 3) ** 2  # variance of movement error use stdev with each measurement var

        # RANSAC parameters
        self.inThreshold = 100  # minimum number of points required for valid line
        self.selectionSize = 100  # ending criteria, once this many points have been fitted end
        self.iterations = 10  # ending criteria, once this many iterations have completed end
        self.distanceThreshold = 0.01  # the points must be within this perpendicular distance from the fitted line

    def start(self):
        self.sensor.start()
        self.odo.start()
        self._start = datetime.now()
        return self

    def stop(self):
        self.sensor.stop()
        self.odo.stop()

    def advanceStep(self):
        self.frame, self.laserScan = self.sensor.getDistanceWithFrame()
        self.stepCount += 1
        self.previousOdometryData = self.currentOdometryData
        self.currentOdometryData = self.odo.getOdometry()
        self.odo.toggleLED()  # this is handy to display progress while not needing to look at terminal

    def EKF(self):
        a = self.currentOdometryData  # for convenience
        b = self.previousOdometryData
        a[2] = self.angleWrap(a[2])
        b[2] = self.angleWrap(b[2])
        timeDelta = (a[5] - b[5]) / 1000

        rot1 = np.arctan2(a[1] - b[1], a[0] - b[0]) - b[2]
        trans = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        rot2 = self.angleWrap(a[2] - b[2] - rot1)

        odometryModel = np.array([trans * np.cos(self.mu[2][0] + self.angleWrap(rot1)),
                                  trans * np.sin(self.mu[2][0] + self.angleWrap(rot1)),
                                  self.angleWrap(rot1 + rot2)])[:, np.newaxis]

        if np.abs(a[4]) < 0.001:
            jacobianG = np.array([[1, 0, -trans * np.sin(a[4] + rot1)],
                                  [0, 1, trans * np.cos(a[4] + rot1)],
                                  [0, 0, 1]])  # jacobian odometry motion model when we are not really rotating
        else:
            r = np.nan_to_num(a[3] / a[4])
            jacobianG = np.array([[1, 0, (r) * (-np.cos(self.mu[2]) + np.cos(self.mu[2] + a[4] * timeDelta))],
                                  [0, 1, (r) * (-np.sin(self.mu[2]) + np.sin(self.mu[2] + a[4] * timeDelta))],
                                  [0, 0, 1]])  # jacobian velocity motion model
        self.prevMu = self.mu
        self.mu = np.vstack((self.prevMu[:3] + odometryModel, self.prevMu[3:]))  # add motion to the state vector but not the landmarks
        self.mu[2] = self.angleWrap(self.mu[2])

        covarianceMu = np.dot(np.dot(jacobianG, self.covariance[:3, :3]), jacobianG.T) + self.R  # calculate GsigmaG' for 3x3 part of covariance
        covarianceMuObservation = np.dot(jacobianG, self.covariance[:3, 3:])  # calculate Gsigma for covariance update on landmarks wrt robot

        self.covariance = np.vstack((np.hstack((covarianceMu, covarianceMuObservation)),
                                     np.hstack((covarianceMuObservation.T, self.covariance[3:, 3:]))))

        observations = self.getFeatures(self.laserScan)

        ObservationWrtRobot = np.column_stack((observations[0, :], self.angleWrap(observations[1, :]))).reshape(-1, 2, 1)  # make 2x1 matrix of all observations in third dimension

        # calculate the expected value of where the landmarks will be relative to robot
        deltaMuLandmarks = np.hstack((
            self.mu[3::2] - self.mu[0],
            self.mu[4::2] - self.mu[1])).T

        if self.mu.shape[0] > 3:
            # calculate the euclidean distance of the landmarks wrt to robot frame
            qLandmarks = np.sqrt(deltaMuLandmarks[0, :] ** 2 + deltaMuLandmarks[1, :] ** 2)
            # convert previous landmarks to distance and angle wrt robot
            zExpectedLandmarksMu = np.column_stack((
                qLandmarks,
                self.angleWrap(np.arctan2(deltaMuLandmarks[1, :], deltaMuLandmarks[0, :]) - self.mu[2]))).reshape(-1, 2, 1)

            # compute observation jacobian for all existing landmarks
            cols = deltaMuLandmarks.shape[1]
            idx = np.column_stack((np.zeros(cols, dtype=np.int), np.ones(cols, dtype=np.int), np.ones(cols, dtype=np.int) * 2, np.arange(cols, dtype=np.int) * 2 + 3, np.arange(cols, dtype=np.int) * 2 + 4)).reshape(cols, 5)
            HalreadyObserved = np.zeros((cols, 2, (cols - 1) * 2 + 5))
            vals = np.array([[-deltaMuLandmarks[0, :] / qLandmarks,
                              -deltaMuLandmarks[1, :] / qLandmarks,
                              np.zeros(cols),
                              deltaMuLandmarks[0, :] / qLandmarks,
                              deltaMuLandmarks[1, :] / qLandmarks],
                             [deltaMuLandmarks[1, :] / (qLandmarks ** 2),
                              -deltaMuLandmarks[0, :] / (qLandmarks ** 2),
                              -np.ones(cols),
                              -deltaMuLandmarks[1, :] / (qLandmarks ** 2),
                              deltaMuLandmarks[0, :] / (qLandmarks ** 2)]])

            HalreadyObserved[np.arange(cols)[:, None], :, idx] = vals.T

            # compute new covariance of landmark relative to robot pose
            phiObs = np.linalg.inv(np.matmul(np.matmul(HalreadyObserved, self.covariance), HalreadyObserved.transpose(0, 2, 1)) + self.Q)
            # find the expected difference of already observed landmarks and the new observations
            zexpectedDifference = (ObservationWrtRobot[:, np.newaxis] - zExpectedLandmarksMu)
            # find the mahalanobis distance of the landmarks in the state matrix
            piObs = np.matmul(np.matmul(zexpectedDifference.transpose(0, 1, 3, 2), phiObs), zexpectedDifference).reshape(ObservationWrtRobot.shape[0], int((self.mu.shape[0] - 3) / 2))
            if np.where(piObs < 0)[0].size > 0:  # make sure that we dont get broken results
                self.logger.error("Non-positive distance for euclidean distance.")
                raise Exception('Non-positive distance for euclidean distance.')
            if piObs.shape[0] > 0:  # we got some observations
                matchedMins, unmatchedMins = self.getMinIndex(np.flipud(piObs))
                NewLandmarks = unmatchedMins[np.where(piObs[unmatchedMins] > self.alpha)[0], 0]  # make sure that the landmarks wont interfere with the other landmarks below the threshold
                NewLandmarks = np.unique(np.append(NewLandmarks, np.where(piObs[matchedMins[:, 0], matchedMins[:, 1]] > self.alpha)[0]))  # add new landmarks from the matched landmarks if they are above threshold
                matchedMins = np.delete(matchedMins, np.where(piObs[matchedMins[:, 0], matchedMins[:, 1]] > self.alpha)[0], axis=0)  # remove the landmarks from the matched landmarks if they are above threshold

                if ObservationWrtRobot[NewLandmarks].shape[0] > 0:  # if we have new landmarks, add them
                    for i in ObservationWrtRobot[NewLandmarks]:
                        self.addLandmark(i)

                i = matchedMins[:, 1]  # mu id  replace i and k with np.array([0,1,2]) for known correspondence
                k = matchedMins[:, 0]  # obs id
                if matchedMins.shape[0] > 0:
                    self.logger.info("observation " + str(k) + " matching landmark id:" + str(i))
                    paddedInvObservationJac = np.dstack((HalreadyObserved[i], np.zeros((i.size, 2, (self.covariance.shape[0] - HalreadyObserved[i].shape[2])))))
                    kalmanGain = np.matmul(np.matmul(self.covariance, paddedInvObservationJac.transpose(0, 2, 1)), phiObs[i])
                    self.mu = self.mu + np.sum(np.matmul(kalmanGain, zexpectedDifference[k, i]), axis=0)
                    self.covariance = np.dot((np.eye(self.covariance.shape[0]) - np.sum(np.matmul(kalmanGain, paddedInvObservationJac), axis=0)), self.covariance)

        if ObservationWrtRobot.shape[0] > 0 and self.mu.shape[0] == 3:
            self.addLandmark(ObservationWrtRobot[0])  # blindly add our first landmark

    def angleWrap(self, angle):
        wrapped = angle - (np.ceil((angle + np.pi) / (2 * np.pi))) * 2 * np.pi  # [-Pi;Pi)
        return wrapped

    def addLandmark(self, obs):
        """Require observation distance and phi wrt robot."""
        size = self.mu.shape[0]
        x = self.mu[0] + obs[0, 0] * np.cos(obs[1, 0] + self.mu[2])
        y = self.mu[1] + obs[0, 0] * np.sin(obs[1, 0] + self.mu[2])
        insideX, _ = np.where(np.logical_and(x + 0.1 > self.mu[3::2], self.mu[3::2] > x - 0.1))
        insideY, _ = np.where(np.logical_and(y + 0.1 > self.mu[4::2], self.mu[4::2] > y - 0.1))
        if(insideX.shape[0] > 0 and insideY.shape[0] > 0):
            print("could not add landmark, too close X: " + str(insideX.shape[0]) + " Y: " + str(insideY.shape[0]))
            return
        self.logger.info("added landmark ID:" + str(int((size - 3) / 2)) + " at (" + str(x[0]) + ", " + str(y[0]) + ")")
        self.mu = np.vstack((self.mu, x, y))

        LandmarksJacobian1 = np.array([[1, 0, -obs[0, 0] * np.sin(self.mu[2] + obs[1, 0])],
                                       [0, 1, obs[0, 0] * np.cos(self.mu[2] + obs[1, 0])]])

        LandmarksJacobian2 = np.array([[np.cos(self.mu[2] + obs[1, 0])[0], -obs[0, 0] * np.sin(self.mu[2] + obs[1, 0])[0]],
                                       [np.sin(self.mu[2] + obs[1, 0])[0], obs[0, 0] * np.cos(self.mu[2] + obs[1, 0])[0]]])

        M = np.vstack((np.hstack((np.eye(size), np.zeros((size, 2)))), np.hstack((LandmarksJacobian1, np.zeros((2, size - 3)), LandmarksJacobian2))))
        self.covariance = np.dot(np.dot(M, block_diag(self.covariance, self.Q)), M.T)

    def getMinIndex(self, a):
        """Associate a min value by removing both the column and row looking at each column iteratively. It then finds values which are unassoicated to a mu."""
        l = np.flipud(a.copy())
        k = np.empty((0, 2), int)
        g = np.empty((0, 2), int)
        for i in np.arange(l.shape[1]):
            d = np.unravel_index(l.argmin(), l.shape)
            k = np.vstack((k, np.array([d[0], d[1]])))
            l[:, d[1]] = np.inf
            l[d[0], :] = np.inf
            if(np.all(np.isinf(l))):
                break
        if a.shape[0] > a.shape[1]:
            m = np.flipud(a.copy())
            m[k[:, 0], :] = np.inf
            for i in np.arange(m.shape[0]):
                d = np.unravel_index(m.argmin(), m.shape)
                g = np.vstack((g, np.array([d[0], d[1]])))
                m[d[0], :] = np.inf
                if(np.all(np.isinf(m))):
                    break
        return k, g

    def getFeatures(self, laserLines):
        """RANSAC implementation."""
        selection = np.arange(laserLines.shape[0])
        dataList = np.zeros([0, 3], dtype=np.float)
        iterations = 0
        while(selection.size > self.selectionSize and iterations < self.iterations):
            choice = np.sort(np.random.choice(selection, 2))
            point = np.append(np.where(selection == choice[0]), np.where(selection == choice[1]))
            with np.errstate(all='ignore'):
                m = (laserLines[point][1, 1] - laserLines[point][0, 1]) / (laserLines[point][1, 0] - laserLines[point][0, 0])
            c = laserLines[point][0, 1] - m * laserLines[point][0, 0]
            distance = np.abs(m * laserLines[:, 0] - laserLines[:, 1] + c) / np.sqrt(m ** 2 + c ** 2)
            inThreshold = np.where(distance < self.distanceThreshold)
            iterations += 1
            if inThreshold[0].size > self.inThreshold:
                selection = np.delete(selection, inThreshold[0])
                dataList = np.vstack((dataList, np.array([m, c, inThreshold[0].size])))
                laserLines = np.delete(laserLines, inThreshold[0], axis=0)
                iterations -= 1
        c = dataList[:, 1][:, np.newaxis]
        m = dataList[:, 0][:, np.newaxis]
        with np.errstate(all='ignore'):
            xi = (c - c.T) / (m.T - m)
            yi = xi * m + c
        searchAngles = np.abs(np.triu(np.arctan(m.T) - np.arctan(m), 1))
        # this is like the hough transform if there is a angle difference of
        # atleast 25 degrees
        noAngle = np.where(~((searchAngles > np.pi * 40 / 180)))
        # set this to a range where we would not actually be able to see
        xi[noAngle] = 6
        yi[noAngle] = 6
        up = np.triu_indices(xi.shape[0], 1)
        # get indicies of intersections which are out of our sight
        removables = np.unique(np.append(np.append(np.where(np.abs(xi[up]) > 1), np.where(yi[up] > 1)), np.where(yi[up] <= 0)))
        xi = np.delete(xi[up], removables)
        yi = np.delete(yi[up], removables)
        d = np.vstack(((np.sqrt(xi ** 2 + yi ** 2), np.arctan2(yi, xi)), xi, yi))
        return d

    def sendData(self, object):
        """Compress data and send it."""
        pickledObject = pickle.dumps(object, -1)
        compressed = zlib.compress(pickledObject)
        try:
            return self.dataSocket.send(compressed, zmq.NOBLOCK)
        except zmq.Again:
            self.logger.error("Could not send data, is the host viewer connected?")
            self.sendError += 1
            if self.sendError > 3:
                raise Exception('Could not reach host viewer.')

    def sendUncompressed(self, object):
        """Compress data and send it."""
        pickledObject = pickle.dumps(object, -1)
        try:
            return self.dataSocket.send(pickledObject, zmq.NOBLOCK)
        except zmq.Again:
            self.logger.error("Could not send data, is the host viewer connected?")
            self.sendError += 1
            if self.sendError > 3:
                raise Exception('Could not reach host viewer.')

    def sendAll(self):
        """Send all the required data for visual display to remote server."""
        # self.packetnumber += 1
        # packets = round(self.packetnumber / (datetime.now() - self.start).total_seconds(), 2)
        self.sendUncompressed(self.stepCount)
        self.sendUncompressed(self.mu)
        self.sendUncompressed(self.covariance)
        self.sendUncompressed(self.currentOdometryData[:2])  # so we can get predicted vs odometry
        self.sendUncompressed(self.frame)  # send the frame and do the post processing on the frame at the receiver to save bandwidth
        #self.sendUncompressed(packets)

__name__ = '__main__'
slam = SLAM()
slam.start()
while True:
    try:
        slam.advanceStep()  # get required data for next iteration
        slam.EKF()  # compute ekf slam
        slam.sendAll()  # send all the data to the listener
    except (KeyboardInterrupt, SystemExit, Exception):
        slam.stop()
        print("Quitting..")
        break

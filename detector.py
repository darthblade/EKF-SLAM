import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import datetime
import logging
import glob
import time
import os


class detector:

    def __init__(self, resolution=(640, 400), framerate=32):  # (1640,1232)

        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.shutter_speed = 9000
        self.camera.iso = 1600

        self.rangeViewMaxDist = 0.7
        self.rangeViewMinDist = 0.1

        self.prevCenterPixel = 0  # this is for callibration purposes

        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format="bgr", use_video_port=True)
        self.frame = None
        self._start = datetime.datetime.now()
        self._framecount = 0
        self.numberOfCallibrationImages = 20
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)
        np.seterr(all='ignore')  # suppress devide by zero warning

        self.callibrateOnStart = False

        if len(glob.glob('callibrationData/*.jpg')) >= self.numberOfCallibrationImages:
            self.logger.debug("Found %s callibration images..", len(
                glob.glob('callibrationData/*.jpg')))
            testImage = cv2.imread(glob.glob('callibrationData/*.jpg')[0])
            if testImage.shape[0] != self.camera.resolution[1] or testImage.shape[1] != self.camera.resolution[0]:
                self.logger.info("Imagedata mismatch->(%s,%s)!=(%s,%s): Callibration required!", testImage.shape[
                                 0], self.camera.resolution[1], testImage.shape[1], self.camera.resolution[0])
                self.callibrateOnStart = True
            if len(glob.glob('callibrationData/*.npz')) is 2 and self.callibrateOnStart is False:
                cameraCallibrationData = glob.glob('callibrationData/*.npz')
                self.logger.debug(
                    "Applying callibration file '%s' to camera.", cameraCallibrationData[0])
                with np.load(cameraCallibrationData[0]) as data:
                    self.mtx = data['mtx']
                    self.dist = data['dist']
                w, h = self.camera.resolution
                self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, (w, h), 1, (w, h))
                self.camCalx, self.camCaly, self.camCalw, self.camCalh = self.roi
                self.camCalhDiv2 = np.floor(self.camCalh / 2)
                self.camEndx = self.camCalx + self.camCalw
                self.camEndy = self.camCalh + self.camCaly
            elif self.callibrateOnStart is False:
                self.logger.debug(
                    "Camera callibration data not found. Callibrating with available images.")
                images = glob.glob('callibrationData/*.jpg')
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # prepare object points, like (0,0,0), (1,0,0), (2,0,0)
                # ....,(6,5,0)
                objp = np.zeros((6 * 8, 3), np.float32)
                objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
                # Arrays to store object points and image points from all the
                # images.
                objpoints = []  # 3d point in real world space
                imgpoints = []  # 2d points in image plane.

                for fname in images:
                    img = cv2.imread(fname)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray, (8, 6), None)

                    if ret is True:
                        objpoints.append(objp)

                        corners2 = cv2.cornerSubPix(
                            gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners2)

                        # Draw and display the corners
                        #img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
                        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                            objpoints, imgpoints, gray.shape[::-1], None, None)
                        w, h = self.camera.resolution
                        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                            self.mtx, self.dist, (w, h), 1, (w, h))
                        self.camCalx, self.camCaly, self.camCalw, self.camCalh = self.roi
                        self.camCalhDiv2 = np.floor(self.camCalh / 2)
                        self.camEndx = self.camCalx + self.camCalw
                        self.camEndy = self.camCalh + self.camCaly
                        np.savez("callibrationData/camera", ret=self.ret, mtx=self.mtx,
                                 dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
                        self.logger.info("Callibrating with image %s", fname)
                self.logger.info("Callibration completed.")
        else:
            self.logger.warning("Could not find camera callibration data.")
            self.callibrateOnStart = True

    def update(self):
        """Update is called by the thread each time a frame is available. It is in a thread because it is an IO bound process."""
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)
            if self.threadRun is False:
                break

    def start(self):
        """Start the thread and deamonize it."""
        self.threadRun = True
        frameUpdate = Thread(target=self.update, args=())
        frameUpdate.daemon = True
        frameUpdate.start()
        time.sleep(1)
        if self.callibrateOnStart is True:
            self.callibrate(True)
        self.setupLookupTable()
        self.logger.info("detector started")
        return self

    def stop(self):
        """Stop the thread from running and exit cleanly."""
        self.threadRun = False
        self.logger.info("detector stopped")
        return self

    def getFrameUncallibrated(self):
        """Return an uncalibrated frame. This should only be called for calibration purposes."""
        return self.frame

    def getFrame(self):
        """Return undistorted camera frame from calibration matrix."""
        return cv2.undistort(self.frame,
                             self.mtx, self.dist, None, self.newcameramtx)[self.camCaly:self.camEndy, self.camCalx:self.camEndx]

    def callibrate(self, forcefullCallibration=False):
        """Calibrate the sensor, by holding a 8x6 checker box pattern in different orientations until x frames are captured. With the frames a calibration matrix is generated."""
        images = glob.glob('callibrationData/*.jpg')
        imageNumber = len(images)
        if imageNumber < self.numberOfCallibrationImages:
            self.logger.warning("Callibration requested:\nFound %s of %s images.",
                                imageNumber, self.numberOfCallibrationImages)
        if forcefullCallibration:
            for f in glob.glob("callibrationData/*.*"):
                self.logger.info("Deleting %s", f)
                os.remove(f)
            images = glob.glob('callibrationData/*.jpg')
            imageNumber = len(images)
        self.logger.info(
            "\n\nPlease hold a chessboard with at least 8 by 6 blocks.\nCallibrating..")
        while imageNumber < self.numberOfCallibrationImages:
            image = self.getFrameUncallibrated()
            ret, corners = cv2.findChessboardCorners(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (8, 6), None)
            if ret is True:
                self.logger.info("Got callibration image: %s", (imageNumber))
                filename = str(imageNumber) + '.jpg'
                cv2.imwrite("callibrationData/" + filename, image)
                imageNumber += 1

    def getDistanceWithFrame(self):
        """Return frame and map touple (x,y)."""
        a = self.getFrame()[:, :, 2]
        data = cv2.medianBlur(cv2.equalizeHist(a[int(np.floor(a.shape[0] / 2)):, :]), 3)
        distanceArray = np.argmax(data, axis=0)
        resultingMap = self.lookupArray[distanceArray.astype(np.integer), np.arange(distanceArray.size)]
        #delete the values which are out of the sensors capability
        resultingMap = np.delete(resultingMap, np.where(np.logical_or(resultingMap[:, 0] > self.rangeViewMaxDist, resultingMap[:, 0] < self.rangeViewMinDist)), axis=0)
        return a, resultingMap

    def setupLookupTable(self):
        """Set up lookup table for pixel to distance relationship."""
        # y, x accessed by row then column
        a = np.zeros((int(np.floor(self.camCalh / 2)), self.camCalw), dtype=(np.float, 2))
        calculateMatrix = False
        try:
            with np.load('callibrationData/distanceToCoordinateMatrix.npz') as data:
                a = data['array']
                if a.shape[0] != np.floor(self.camCalh / 2) or (a.shape[1]-1) != self.camCalw:
                    self.logger.info("Existing distanceToCoordinate Matrix does not match shape of camera!")
                    self.logger.debug(str(a.shape[0]) + "!=" + str(np.floor(self.camCalh / 2)) + " or " + str(a.shape[1]) + "!=" + str(self.camCalw))
                    calculateMatrix = True
        except IOError:
            calculateMatrix = True
        if calculateMatrix:
            self.logger.info("Calculating new distanceToCoordinate Matrix.")
            col = -np.floor(self.camCalw / 2)
            for x in range(0, self.camCalw):
                row = 0
                for y in range(0, int(np.floor(self.camCalh / 2))):  # td
                    yp = 1 / (0.0524 * (row - 22) + 0.0412)  # see documentation on how to find these values
                    xp = 0.025 * col / (0.7731 * (row) + 0.3868)
                    # a[y, x][0] = xp
                    # a[y, x][1] = yp
                    a[y, x][0] = yp  #rotate around 0,0 for -pi/2
                    a[y, x][1] = -xp
                    row += 1
                col += 1
            np.savez('callibrationData/distanceToCoordinateMatrix', array=a)
            self.logger.info("Done.")
        self.lookupArray = a

    def getAverageCenterPixel(self):
        """Return 50px by height frame around centre with whitened lines around center pixels as well as detected max intensity centerPixel."""
        k, d = self.getDistanceArrayWithFrame()
        k[:, int(self.camCalw / 2) - 1] = 255
        k[:, int(self.camCalw / 2) + 1] = 255
        k = k[:, (int(self.camCalw / 2) - 50):(int(self.camCalw / 2) + 50)]
        self.prevCenterPixel = int((self.prevCenterPixel + d[int(self.camCalw / 2)]) / 2)  # average over time
        return k, d[int(self.camCalw / 2)]

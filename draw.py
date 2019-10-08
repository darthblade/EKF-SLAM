import matplotlib.pylab as plt
from matplotlib.patches import Ellipse, RegularPolygon
import numpy as np
from datetime import datetime

class paint:

    def start(self, name, viewPort=None, output=False):

        self.output = False
        self.start = datetime.now()
        self.frame=0
        plt.ion()
        self.fig = plt.figure(name)
        if viewPort is True:
            self.image = plt.subplot(221)
            self.image2laser = plt.subplot(223)
            self.map = plt.subplot(122)
            self.image.set_title('Raw image from sensor')
            self.image2laser.set_title('Perceived laser height')
            self.map.set_title('Map view')
            self.map.yaxis.tick_right()
            self.viewPort = True
        else:
            self.map = plt.subplot(111)
            self.map.set_title('Map view')
            self.viewPort = False
        if output is True:
            self.output = True
            self.outputCounter = 0
        self.draw()

    def draw(self):
        self.frame += 1
        self.fps = round(self.frame / (datetime.now() - self.start).total_seconds(), 2)
        self.fig.canvas.draw()

    def clear(self):
        self.fig.canvas.flush_events()
        if self.output is True:
            self.fig.savefig('.\\Results\\' + str(self.outputCounter) + '.png')
            self.outputCounter += 1
        if self.viewPort is True:
            self.image.cla()
            self.image2laser.cla()
            self.map.cla()
        else:
            self.map.cla()

    def stop(self, name):
        self.map.cla()
        plt.ioff()
        plt.close(name)

    def show(self):
        plt.show()

    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def drawRobot(self, pos, cov, colour="red"):
        x = pos[0]
        y = pos[1]
        o = pos[2]
        xshift = np.cos(o) * 0.1 + x
        yshift = np.sin(o) * 0.1 + y
        self.map.scatter(
            (x, xshift),
            (y, yshift),
            s=4, marker="x", color=colour)
        p = RegularPolygon((x, y), 3, 0.1, orientation=o + 0.5, fill=False, edgecolor=colour)
        vals, vecs = self.eigsorted(cov[0:2, 0:2])
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        e = Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=0.5, color='gray', zorder=50, fill=False)
        self.map.add_artist(e)
        self.map.add_patch(p)
        self.map.set_title('Map view FPS: ' + str(self.fps))
        self.map.set_ylabel('Y coordinate (m)')
        self.map.set_xlabel('X coordinate (m)')

    def drawAll(self, pos, cov, image, distanceAsImage, distance):
        self.drawFeatures(pos, cov)
        self.drawRobot(pos, cov)

        self.image.imshow(image)
        self.image2laser.set_ylim([0, image.shape[0] / 2])
        self.image2laser.invert_yaxis()
        self.image2laser.scatter(np.arange(distanceAsImage.size), distanceAsImage[::-1], s=4, marker=".", color='orange')
        self.map.axis('equal')
        x = (distance[:, 0] * np.cos(pos[2]) - distance[:, 1] * np.sin(pos[2]) + pos[0])
        y = (distance[:, 0] * np.sin(pos[2]) + distance[:, 1] * np.cos(pos[2]) + pos[1])
        self.map.scatter(x, y, s=4, marker=".")
        self.image.set_title('Raw image from sensor')
        self.image2laser.set_title('Perceived laser height')
        self.map.set_title('Map view FPS: ' + str(self.fps))
        self.map.yaxis.tick_right()
        self.image.set_ylabel('Pixel y coordinate')
        self.image.set_ylabel('Pixel x coordintate')
        self.image2laser.set_ylabel('extracted pixel y coordinate')
        self.image2laser.set_ylabel('extracted pixel x coordintate')

    def drawMap(self, pos, cov, distance):
        self.drawFeatures(pos, cov)
        self.drawRobot(pos, cov)
        x = (distance[:, 0] * np.cos(pos[2]) - distance[:, 1] * np.sin(pos[2]) + pos[0])
        y = (distance[:, 0] * np.sin(pos[2]) + distance[:, 1] * np.cos(pos[2]) + pos[1])
        self.map.scatter(x, y, s=4, marker=".")
        self.map.set_title('Map view FPS: ' + str(self.fps))
        self.map.yaxis.tick_right()

    def drawFeatures(self, pos, cov, colour="green"):
        featureCov = np.array([cov[i:i + 2, i:i + 2] for i in range(3, len(cov), 2)])
        x = pos[3::2]
        y = pos[4::2]
        index = 0
        for i in featureCov:
            self.map.scatter(x[index][0], y[index][0],
                             s=4, marker="x", color=colour)
            self.map.text(x[index][0], y[index][0], u'ID:' + str(index), fontsize=9)
            vals, vecs = self.eigsorted(i)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)
            e = Ellipse(xy=[x[index][0], y[index][0]], width=width, height=height, angle=theta,
                        alpha=0.5, color='gray', zorder=50, fill=False)
            self.map.add_artist(e)
            index += 1
        self.map.axis('equal')

    def drawPath(self, x, y, colour="blue"):
        self.map.scatter(x, y, s=4, marker=".", color=colour)

    def drawView(self, image, distanceAsImage, distance, pos):
        self.image.imshow(image)
        self.image2laser.set_ylim([0, image.shape[0] / 2])
        self.image2laser.invert_yaxis()
        self.image2laser.scatter(np.arange(distanceAsImage.size), distanceAsImage[::-1], s=4, marker=".", color='orange')
        self.map.axis('equal')
        x = (distance[:, 0] * np.cos(pos[2]) - distance[:, 1] * np.sin(pos[2]) + pos[0])
        y = (distance[:, 0] * np.sin(pos[2]) + distance[:, 1] * np.cos(pos[2]) + pos[1])
        self.map.scatter(x, y, s=4, marker=".")
        self.image.set_title('Raw image from sensor')
        self.image2laser.set_title('Perceived laser height')
        self.map.yaxis.tick_right()
        self.image.set_ylabel('Pixel y coordinate')
        self.image.set_ylabel('Pixel x coordintate')
        self.image2laser.set_ylabel('extracted pixel y coordinate')
        self.image2laser.set_ylabel('extracted pixel x coordintate')

    def drawInterceptsOnView(self, c, i, color="black"):
        if c.shape[0] > 1:
            x = np.array([-0.5, 0.5])[:, np.newaxis]
            y = c[1:, 0] * x + c[1:, 1]
            self.map.plot(x, y, c=color)
        self.map.scatter(i[0, :], i[1, :], c="red")

    def drawDetectedFeaturesOnView(self, d, pos, i, color="black"):
        # for t in np.arange(i.shape[0]):
        #     x=np.array([-0.5,0.5])
        #     y=i[t,0]*x+i[t,1]
        #     #print(i[t,0])
        #     self.map.plot(x,y)
        # self.map.set_xlim([-0.5,0.5])
        # self.map.set_ylim([0,0.8])
        fx = d[0, :] * np.cos(d[1, :] + pos[2]) + pos[0]
        fy = d[0, :] * np.sin(d[1, :] + pos[2]) + pos[1]
        # fx = d[0, :] * np.cos(d[1, :])
        # fy = d[0, :] * np.sin(d[1, :])
        self.map.scatter(fx, fy, c=color)

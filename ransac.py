import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, RegularPolygon

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


points=20
phi=np.arange(-np.pi*0.25,np.pi*0.25,0.5*np.pi/(points))
r=1/np.cos(phi)
x=np.ones(points)
y=r*np.sin(phi)

var_phi=np.var(phi)
var_r=np.var(r)

Q=np.vstack((var_phi*r**2*np.sin(phi)**2+var_r*np.cos(phi)**2,-var_phi*r**2/2*np.sin(2*phi)+var_r*np.sin(2*phi)/2,
    -var_phi*r**2/2*np.sin(2*phi)+var_r*np.sin(2*phi)/2,var_phi*r**2*np.cos(phi)**2+var_r*np.sin(phi)**2)).T.reshape(-1,2,2)


ax = plt.subplot(111)
index=0
for i in Q:
    vals, vecs = eigsorted(i)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 0.2*np.sqrt(vals)
    ell = Ellipse(xy=(x[index], y[index]),
                  width=w, height=h,
                  angle=theta, color='black')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    index+=1


plt.scatter(x,y)
plt.scatter(0,0)
plt.show()


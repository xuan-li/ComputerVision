import numpy as np
import pandas as pd
from pandas import plotting
from matplotlib import pyplot as plt

#df = pd.read_csv("HW3_result.csv")
# print data

CAM = pd.read_csv('output_camshift.txt')
CAM.columns = ['index', 'x', 'y']

Kalman = pd.read_csv('output_kalman.txt')
Kalman.columns = ['index', 'x', 'y']

particle = pd.read_csv('output_particle.txt')
particle.columns = ['index', 'x', 'y']

opticle = pd.read_csv('output_of.txt')
opticle.columns = ['index', 'x', 'y']


plt.scatter(x='x', y='y', data=CAM)
plt.scatter(x='x', y='y', data=Kalman)
plt.scatter(x='x', y='y', data=particle)
plt.scatter(x='x', y='y', data=opticle)
plt.legend(['CAM', 'Kalman', 'Particle', 'Opticle'])
#plt.legend(['CAM', 'Particle', 'Opticle'])
plt.show()
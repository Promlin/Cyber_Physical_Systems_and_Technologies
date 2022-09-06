import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

data_read = np.genfromtxt('testLab1Var6', delimiter=',')
time = data_read[:, 0]
time = time[:, np.newaxis]
current = data_read[:, 1]
current = current[:, np.newaxis]
voltage = data_read[:, 2]
voltage = voltage[:, np.newaxis]

fig, (ay1, ay2) = plt.subplots(2, 1, sharex=True)
ay1.plot(time, voltage)
T_per = 0.1
ay1.plot(time[time < 2 * T_per], voltage[time < 2 * T_per])

plt.show()


from matplotlib import pyplot as plt
import numpy as np
from antenna_environment import AntennaEnvironment


env = AntennaEnvironment(distance_m=100)

angles = np.arange(0, 181, 1)
rssi = [env._compute_rssi(rx_theta=a, rx_phi=0, samples=20) for a in angles]

plt.plot(angles, rssi)
plt.xlabel("Azimuth (deg)")
plt.ylabel("RSSI (dBm)")
plt.show()

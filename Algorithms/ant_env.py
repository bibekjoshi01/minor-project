import numpy as np
import matplotlib.pyplot as plt

# Environment function (same as defined)
def rssi_environment(angle,
                     tx_angle=90,
                     base_rssi=-72,
                     g_max=10.5,
                     sigma=17,
                     noise_std=2.0):

    main = g_max * np.exp(-0.5 * ((angle - tx_angle) / sigma)**2)
    side1 = 0.2 * g_max * np.exp(-0.5 * ((angle - (tx_angle + 60)) / (1.8*sigma))**2)
    side2 = 0.2 * g_max * np.exp(-0.5 * ((angle - (tx_angle - 60)) / (1.8*sigma))**2)

    noise = np.random.normal(0, noise_std)

    return base_rssi + main + side1 + side2 + noise

# Generate angles and RSSI samples
angles = np.arange(0, 181, 1)
rssi_values = [rssi_environment(a) for a in angles]

# Plot
plt.figure()
plt.plot(angles, rssi_values)
plt.xlabel("Antenna Angle (degrees)")
plt.ylabel("RSSI (dBm)")
plt.title("Simulated RSSI vs Antenna Angle (10.5 dBi Yagi)")
plt.show()

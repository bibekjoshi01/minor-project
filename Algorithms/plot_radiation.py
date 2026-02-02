import matplotlib.pyplot as plt
import numpy as np

from rssi_sim import synthetic_yagi_pattern_deg

# --- Angle sweeps ---
azimuth_deg = np.linspace(-180, 180, 720)  # horizontal
elevation_deg = np.linspace(-90, 90, 360)  # vertical

# Compute patterns
az_gain = synthetic_yagi_pattern_deg(azimuth_deg)
el_gain = synthetic_yagi_pattern_deg(elevation_deg)

# Normalize for polar plotting (0 dB = max)
az_norm = az_gain - np.max(az_gain)
el_norm = el_gain - np.max(el_gain)

# --- Plot Azimuth Pattern ---
plt.figure(figsize=(8, 6))
ax1 = plt.subplot(121, projection="polar")
ax1.plot(np.deg2rad(azimuth_deg), az_norm, color="b", lw=2)
ax1.set_theta_zero_location("N")  # boresight at top
ax1.set_theta_direction(-1)  # clockwise
ax1.set_rlim(-40, 0)
ax1.set_title("Azimuth Pattern (Horizontal Cut)")

# --- Plot Elevation Pattern ---
ax2 = plt.subplot(122, projection="polar")
ax2.plot(np.deg2rad(elevation_deg), el_norm, color="r", lw=2)
ax2.set_theta_zero_location("N")  # boresight at top
ax2.set_theta_direction(-1)  # clockwise
ax2.set_rlim(-40, 0)
ax2.set_title("Elevation Pattern (Vertical Cut)")

plt.tight_layout()
plt.show()

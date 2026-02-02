import numpy as np
import matplotlib.pyplot as plt
from rssi_sim import AntennaPattern, AntennaSimulation

tx_pat = AntennaPattern(use_synthetic=True, g_max_dbi=10.5, beamwidth_deg=20.0)
rx_pat = AntennaPattern(use_synthetic=True, g_max_dbi=10.5, beamwidth_deg=20.0)

sim = AntennaSimulation(
    freq_ghz=2.45,
    tx_power_dbm=20.0,
    tx_pos=(0.0, 0.0, 1.0),
    rx_pos=(10.0, 0.0, 1.0),  # 40 meters separation
    tx_az_deg=0.0,
    tx_el_deg=0.0,
    rx_az_deg=180.0,
    rx_el_deg=0.0,
    tx_pattern=tx_pat,
    rx_pattern=rx_pat,
    path_loss_exponent=2.2,
    shadow_std_db=3.5,
    fastfade_std_db=1.5,
    samples_per_point=20,
    seed=42,
)

# Sweep azimuth (0..360) at tilt 0 degrees
azs, means, stds = sim.sweep_azimuth(np.arange(0, 360, 2.0), tilt_deg=0.0)

# Plot RSSI vs azimuth
plt.figure(figsize=(9, 4))
plt.plot(azs, means, "-o", markersize=3)
plt.fill_between(azs, means - stds, means + stds, color="gray", alpha=0.25)
plt.xlabel("Rx azimuth (deg) — 0° = pointing to Tx along +X")
plt.ylabel("RSSI (dBm)")
plt.title("RSSI vs Rx azimuth (tilt = 0°)")
plt.grid(True)
plt.xlim(0, 360)
plt.show()

# Polar plot (optional)
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
theta_radians = np.deg2rad(azs)
ax.plot(theta_radians, means)
ax.set_title("Polar RSSI vs Azimuth")
ax.set_theta_zero_location("E")  # 0 deg = +X on plot
ax.set_theta_direction(-1)  # clockwise
plt.show()

# Example: sweep elevation at best azimuth
best_idx = np.argmax(means)
best_az = azs[best_idx]
elevs, elev_means, elev_stds = sim.sweep_elevation(
    np.arange(-30, 31, 1.0), azimuth_deg=best_az
)

plt.figure(figsize=(8, 4))
plt.plot(elevs, elev_means, "-o", markersize=3)
plt.fill_between(elevs, elev_means - elev_stds, elev_means + elev_stds, alpha=0.25)
plt.xlabel("Rx elevation (tilt) deg (positive = up)")
plt.ylabel("RSSI (dBm)")
plt.title(f"RSSI vs elevation at azimuth={best_az:.1f}° (best az)")
plt.grid(True)
plt.show()

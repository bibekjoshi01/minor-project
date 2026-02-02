import numpy as np
import matplotlib.pyplot as plt

from baseline_algos.exhaustive_scan import ExhaustiveScan2D
from ant_env.environment import AntennaEnvironment

env = AntennaEnvironment(distance_m=100)

scanner = ExhaustiveScan2D(
    env,
    pan_range=(30, 150),
    tilt_range=(60, 120),
    pan_step=10,
    tilt_step=5,
    fine_step=2,
    samples_per_point=20,
    settle_time=0.0,
)

best_rssi, best_point = scanner.run_scan()

print("Best RSSI:", best_rssi)
print("Best Point:", best_point)

pan_vals, tilt_vals, rssi_grid = scanner.get_rssi_grid()

plt.figure(figsize=(10, 6))

X, Y = np.meshgrid(tilt_vals, pan_vals)
plt.contourf(X, Y, rssi_grid, levels=50)
plt.colorbar(label="RSSI (dBm)")

plt.scatter(
    best_point["tilt"],
    best_point["pan"],
    color="red",
    s=80,
    label=f"Best: {round(best_rssi, 2)}",
)
plt.xlabel("Tilt (deg)")
plt.ylabel("Pan (deg)")
plt.title("RSSI Surface (Exhaustive Scan)")
plt.legend()
plt.show()
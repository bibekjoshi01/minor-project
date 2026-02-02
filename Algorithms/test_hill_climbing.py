import matplotlib.pyplot as plt
from baseline_algos.hill_climbing import HillClimb2D
from ant_env.environment import AntennaEnvironment

env = AntennaEnvironment(distance_m=100)

hc = HillClimb2D(
    env,
    pan_range=(0, 30),
    tilt_range=(60, 120),
    step_size=2,
    samples_per_point=20,
    max_iters=200,
    patience=15,
    init_point=(40, 70),  # deliberately bad start
)

best_rssi, best_point, path = hc.run_scan()

print("Best RSSI:", best_rssi)
print("Best Point:", best_point)

# ---- plot path ----
pan = [p["pan"] for p in path]
tilt = [p["tilt"] for p in path]

plt.figure(figsize=(6, 6))
plt.plot(tilt, pan, "-o")
plt.scatter(best_point["tilt"], best_point["pan"], c="red", s=100)
plt.xlabel("Tilt")
plt.ylabel("Pan")
plt.title("Hill Climbing Path")
plt.show()

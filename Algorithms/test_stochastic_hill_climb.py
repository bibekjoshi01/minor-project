import matplotlib.pyplot as plt
from baseline_algos.stochastic_hill_climb import StochasticHillClimb2D
from ant_env.antenna_environment import AntennaEnvironment

env = AntennaEnvironment(distance_m=100)

shc = StochasticHillClimb2D(
    env,
    pan_range=(30, 150),
    tilt_range=(60, 120),
    step_size=2,
    samples_per_point=20,
    max_iters=200,
    patience=15,
    settle_time=0.0,
    random_jump_prob=0.2,
    init_point=(40, 70),  
)

best_rssi, best_point, path = shc.run_scan()

print("Best RSSI:", best_rssi)
print("Best Point:", best_point)

pan = [p["pan"] for p in path]
tilt = [p["tilt"] for p in path]

plt.figure(figsize=(6, 6))
plt.plot(tilt, pan, "-o")
plt.scatter(best_point["tilt"], best_point["pan"], c="red", s=100, label="Best")
plt.xlabel("Tilt")
plt.ylabel("Pan")
plt.title("Stochastic Hill Climbing Path")
plt.legend()
plt.show()

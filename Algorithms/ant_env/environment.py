from .base import AntennaEnvironmentBase


class AntennaEnvironment(AntennaEnvironmentBase):
    def __init__(self):
        super().__init__()

    def set_orientation(self, pan, tilt):
        return super().set_orientation(pan, tilt)

    def measure_rssi(self, samples=1):
        return super().measure_rssi(samples)

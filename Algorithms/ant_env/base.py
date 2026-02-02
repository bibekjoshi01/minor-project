from abc import ABC, abstractmethod


class AntennaEnvironmentBase(ABC):
    """
    Abstract base class for any antenna environment (simulation or hardware)
    """

    @abstractmethod
    def set_orientation(self, pan: int, tilt: int):
        """
        Rotate the antenna to (pan, tilt)
        Args:
            pan: horizontal angle (deg)
            tilt: vertical angle (deg)
        """
        pass

    @abstractmethod
    def measure_rssi(self, samples: int = 1) -> float:
        """
        Measure RSSI at current orientation.
        Args:
            samples: number of measurements to average
        Returns:
            rssi (float)
        """
        pass

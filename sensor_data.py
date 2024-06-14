import numpy as np
from rich.table import Table
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

class SensorData:
    def __init__(self, data=None):
        if data is None:
            data = [0.0] * 9
        self.magnetometer = data[:3]
        self.gyroscope = data[3:6]
        self.accelerometer = data[6:9]
        self.calibration_data = {'magnetometer': [], 'gyroscope': [], 'accelerometer': []}
        self.calibrated = False
        self.biases = {'magnetometer': [0.0, 0.0, 0.0], 'gyroscope': [0.0, 0.0, 0.0], 'accelerometer': [0.0, 0.0, 9.81]}
        self.collecting_calibration_data = True
        # Initialize UKFs for each axis of each sensor
        self.ukf_magnetometer = [self.init_ukf() for _ in range(3)]
        self.ukf_gyroscope = [self.init_ukf() for _ in range(3)]
        self.ukf_accelerometer = [self.init_ukf() for _ in range(3)]
        print("Sensor data initialized")

    def init_ukf(self):
        def fx(x, dt):
            # State transition function
            return np.array([x[0] + dt*x[1], x[1]])

        def hx(x):
            # Measurement function
            return np.array([x[0]])

        points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0)
        ukf = UKF(dim_x=2, dim_z=1, fx=fx, hx=hx, dt=1.0, points=points)
        ukf.x = np.array([0., 0.])      # Initial state (position and velocity)
        ukf.P *= 0.1                    # Initial state covariance
        ukf.R = np.array([[0.01]])      # Measurement noise
        ukf.Q = np.array([[1e-9, 0.],
                        [0., 1e-9]])  # Process noise
        return ukf

    def calibrate(self):
        # Compute biases as the mean of the collected calibration data
        for sensor in self.calibration_data:
            if sensor != 'magnetometer':  # Skip magnetometer calibration
                self.biases[sensor] = np.mean(self.calibration_data[sensor], axis=0)
        self.biases['accelerometer'][2] -= 9.81  # Adjust for gravity on the Z-axis
        self.calibrated = True
        self.collecting_calibration_data = False
        print("Calibration complete. Biases calculated:", self.biases)

    def update(self, data):
        self.magnetometer = data[:3]
        self.gyroscope = data[3:6]
        self.accelerometer = data[6:9]

        if self.collecting_calibration_data:
            self.calibration_data['magnetometer'].append(self.magnetometer)
            self.calibration_data['gyroscope'].append(self.gyroscope)
            self.calibration_data['accelerometer'].append(self.accelerometer)
            if len(self.calibration_data['magnetometer']) >= 2000:  # Collect data for about 1 minute at 10 Hz
                self.calibrate()
        else:
            # Apply biases to the sensor data
            self.magnetometer = [self.magnetometer[i] for i in range(3)]  # No calibration applied
            self.gyroscope = [self.gyroscope[i] - self.biases['gyroscope'][i] for i in range(3)]
            self.accelerometer = [self.accelerometer[i] - self.biases['accelerometer'][i] for i in range(3)]

    def apply_ukf(self, values, ukf_list):
        filtered_values = []
        for i in range(3):
            ukf = ukf_list[i]
            z = np.array([values[i]])
            ukf.predict()
            ukf.update(z)
            filtered_values.append(ukf.x[0])
        return filtered_values

    def to_table(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Sensor", style="dim", width=15)
        table.add_column("X", justify="right")
        table.add_column("Y", justify="right")
        table.add_column("Z", justify="right")
        table.add_column("UKF X", justify="right")
        table.add_column("UKF Y", justify="right")
        table.add_column("UKF Z", justify="right")

        # Apply UKF to each sensor reading
        filtered_magnetometer_ukf = self.apply_ukf(self.magnetometer, self.ukf_magnetometer)
        filtered_gyroscope_ukf = self.apply_ukf(self.gyroscope, self.ukf_gyroscope)
        filtered_accelerometer_ukf = self.apply_ukf(self.accelerometer, self.ukf_accelerometer)

        table.add_row("Magnetometer", f"{self.magnetometer[0]:.6f}", f"{self.magnetometer[1]:.6f}", f"{self.magnetometer[2]:.6f}",
                      f"{filtered_magnetometer_ukf[0]:.6f}", f"{filtered_magnetometer_ukf[1]:.6f}", f"{filtered_magnetometer_ukf[2]:.6f}")
        table.add_row("Gyroscope", f"{self.gyroscope[0]:.6f}", f"{self.gyroscope[1]:.6f}", f"{self.gyroscope[2]:.6f}",
                      f"{filtered_gyroscope_ukf[0]:.6f}", f"{filtered_gyroscope_ukf[1]:.6f}", f"{filtered_gyroscope_ukf[2]:.6f}")
        table.add_row("Accelerometer", f"{self.accelerometer[0]:.6f}", f"{self.accelerometer[1]:.6f}", f"{self.accelerometer[2]:.6f}",
                      f"{filtered_accelerometer_ukf[0]:.6f}", f"{filtered_accelerometer_ukf[1]:.6f}", f"{filtered_accelerometer_ukf[2]:.6f}")

        return table

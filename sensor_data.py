import numpy as np
import matplotlib.pyplot as plt
from rich.table import Table
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from tqdm import tqdm

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
        self.progress_bar = tqdm(total=10000, desc="Calibrating sensors", unit="samples")
        self.gyro_data = []
        self.accel_data = []
        self.ukf_gyro_data = []
        self.ukf_accel_data = []
        # Initialize UKFs for each axis of each sensor
        self.ukf_magnetometer = [self.init_ukf() for _ in range(3)]
        self.ukf_gyroscope = [self.init_ukf() for _ in range(3)]
        self.ukf_accelerometer = [self.init_ukf() for _ in range(3)]
        print("Sensor data initialized")

    def init_ukf(self):
        def fx(x, dt):
            return np.array([x[0] + dt*x[1], x[1]])

        def hx(x):
            return np.array([x[0]])

        points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0)
        ukf = UKF(dim_x=2, dim_z=1, fx=fx, hx=hx, dt=1.0, points=points)
        ukf.x = np.array([0., 0.])
        ukf.P *= 0.1
        ukf.R = np.array([[0.01]])
        ukf.Q = np.array([[1e-9, 0.],
                          [0., 1e-9]])
        return ukf

    def calibrate(self):
        for sensor in self.calibration_data:
            if sensor != 'magnetometer':
                self.biases[sensor] = np.mean(self.calibration_data[sensor], axis=0)
        self.biases['accelerometer'][2] -= 9.81
        self.calibrated = True
        self.collecting_calibration_data = False
        self.progress_bar.close()
        print("Calibration complete. Biases calculated:", self.biases)

    def update(self, data):
        self.magnetometer = data[:3]
        self.gyroscope = data[3:6]
        self.accelerometer = data[6:9]

        if self.collecting_calibration_data:
            self.calibration_data['magnetometer'].append(self.magnetometer)
            self.calibration_data['gyroscope'].append(self.gyroscope)
            self.calibration_data['accelerometer'].append(self.accelerometer)
            self.progress_bar.update(1)
            if len(self.calibration_data['magnetometer']) >= 10000:
                self.calibrate()
        else:
            self.magnetometer = [self.magnetometer[i] for i in range(3)]
            self.gyroscope = [self.gyroscope[i] - self.biases['gyroscope'][i] for i in range(3)]
            self.accelerometer = [self.accelerometer[i] - self.biases['accelerometer'][i] for i in range(3)]

            self.gyro_data.append(self.gyroscope)
            self.accel_data.append(self.accelerometer)

            filtered_gyro = self.apply_ukf(self.gyroscope, self.ukf_gyroscope)
            filtered_accel = self.apply_ukf(self.accelerometer, self.ukf_accelerometer)

            self.ukf_gyro_data.append(filtered_gyro)
            self.ukf_accel_data.append(filtered_accel)

    def apply_ukf(self, values, ukf_list):
        filtered_values = []
        for i in range(3):
            ukf = ukf_list[i]
            z = np.array([values[i]])
            ukf.predict()
            ukf.update(z)
            filtered_values.append(ukf.x[0])
        return filtered_values

    def plot_data(self):
        raw_gyro = np.array(self.gyro_data)
        raw_accel = np.array(self.accel_data)
        filtered_gyro = np.array(self.ukf_gyro_data)
        filtered_accel = np.array(self.ukf_accel_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(raw_gyro[:, 0], label='Raw Gyro X', color='r')
        axes[0, 0].plot(filtered_gyro[:, 0], label='Filtered Gyro X', color='g')
        axes[0, 0].set_title('Gyroscope X-axis')
        axes[0, 0].legend()
        
        axes[0, 1].plot(raw_accel[:, 0], label='Raw Accel X', color='r')
        axes[0, 1].plot(filtered_accel[:, 0], label='Filtered Accel X', color='g')
        axes[0, 1].set_title('Accelerometer X-axis')
        axes[0, 1].legend()
        
        axes[1, 0].plot(raw_gyro[:, 1], label='Raw Gyro Y', color='r')
        axes[1, 0].plot(filtered_gyro[:, 1], label='Filtered Gyro Y', color='g')
        axes[1, 0].set_title('Gyroscope Y-axis')
        axes[1, 0].legend()
        
        axes[1, 1].plot(raw_accel[:, 1], label='Raw Accel Y', color='r')
        axes[1, 1].plot(filtered_accel[:, 1], label='Filtered Accel Y', color='g')
        axes[1, 1].set_title('Accelerometer Y-axis')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

    def to_table(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Sensor", style="dim", width=15)
        table.add_column("X", justify="right")
        table.add_column("Y", justify="right")
        table.add_column("Z", justify="right")
        table.add_column("UKF X", justify="right")
        table.add_column("UKF Y", justify="right")
        table.add_column("UKF Z", justify="right")

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

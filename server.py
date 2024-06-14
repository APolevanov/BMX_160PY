import socket
import struct
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from sensor_data import SensorData

SERVER_IP = "0.0.0.0"
SERVER_PORT = 12345

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
udp_socket.bind((SERVER_IP, SERVER_PORT))

console = Console()

def create_panel(sensor_data, addr):
    panel = Panel.fit(sensor_data.to_table(), title=f"Client: {addr[0]}:{addr[1]}", border_style="green")
    return panel

console.print(f"Server started at port {SERVER_PORT}", style="bold green")

sensor_data = SensorData()

with Live(console=console, refresh_per_second=2) as live:
    try:
        while True:
            data, addr = udp_socket.recvfrom(1024)
            floats = struct.unpack('9f', data)
            sensor_data.update(floats)
            panel = create_panel(sensor_data, addr)
            live.update(panel)
            response = "Data received".encode('utf-8')
            udp_socket.sendto(response, addr)
    except KeyboardInterrupt:
        console.print("Stopping server and plotting data...", style="bold red")
        sensor_data.plot_data()
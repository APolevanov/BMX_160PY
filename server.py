import socket
import struct
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from sensor_data import SensorData

SERVER_IP = "0.0.0.0"  # Bind to all available interfaces
SERVER_PORT = 12345

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind the socket to the address
udp_socket.bind((SERVER_IP, SERVER_PORT))

console = Console()

def create_panel(sensor_data, addr):
    panel = Panel.fit(sensor_data.to_table(), title=f"Client: {addr[0]}:{addr[1]}", border_style="green")
    return panel

console.print(f"Server started at port {SERVER_PORT}", style="bold green")

sensor_data = SensorData()

with Live(console=console, refresh_per_second=2) as live:
    while True:
        # Receive data from the client
        data, addr = udp_socket.recvfrom(1024)

        # Unpack the data assuming 9 floats (4 bytes each)
        floats = struct.unpack('9f', data)

        # Update sensor data
        sensor_data.update(floats)

        # Update the live display with new data
        panel = create_panel(sensor_data, addr)
        live.update(panel)

        # Send a response to the client
        response = "Data received".encode('utf-8')
        udp_socket.sendto(response, addr)

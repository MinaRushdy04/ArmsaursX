#!/usr/bin/env python3
"""
Robust Serial Servo Control Interface
-----------------------
Controls Arduino servos via serial communication.
Receives joint angles from IK.py via multiprocessing.Queue.
"""

import serial
import time
import sys
import threading
import logging
import platform
import multiprocessing
from typing import Optional

try:
    import serial.tools.list_ports
except ImportError:
    print("❌ Missing pyserial module. Install with `pip install pyserial`.")
    sys.exit(1)

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ---------------- Helper Functions ----------------
def get_default_port():
    return "COM9" if platform.system() == "Windows" else "/dev/ttyUSB0"

def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def select_serial_port(default_port: str) -> str:
    ports = get_serial_ports()
    logging.info(f"Default port: {default_port}")
    if default_port in ports:
        if input(f"Use {default_port}? (Y/n): ").strip().lower() in ['', 'y', 'yes']:
            return default_port
    if not ports:
        port = input("No ports found, enter manually: ").strip()
        return port if port else default_port
    print("Available Serial Ports:")
    for i, p in enumerate(ports, 1):
        print(f"{i}. {p}")
    choice = input(f"Select port number (or Enter for {default_port}): ").strip()
    if choice == "":
        return default_port
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ports):
            return ports[idx]
    except:
        pass
    return choice  # fallback to typed value

# ---------------- Serial Servo Controller ----------------
class RobustSerialServoController:
    def __init__(self, port: str, baudrate: int = 9600, angle_queue: multiprocessing.Queue = None):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.angle_queue = angle_queue if angle_queue else multiprocessing.Queue()
        self.connected = False
        self.running = False
        self.thread = None

        # Servo ranges (degrees)
        self.servo_ranges = {1: (0, 180), 2: (0, 180), 3: (0, 180)}
        self.current_angles = {1: 0, 2: 0, 3: 0}

    def connect(self):
        try:
            logging.info(f"🔄 Connecting to Arduino on {self.port}...")
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=2, write_timeout=2)
            time.sleep(2)
            self.connected = True
            logging.info(f"✅ Connected to Arduino on {self.port}")
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")

    def disconnect(self):
        if self.serial_connection and self.connected:
            try:
                self.serial_connection.close()
                self.connected = False
                logging.info("🔌 Disconnected from Arduino")
            except Exception as e:
                logging.warning(f"⚠️ Error closing serial: {e}")

    def send_angle(self, servo_id: int, angle: float):
        if servo_id not in self.servo_ranges:
            logging.error(f"❌ Invalid servo id: {servo_id}")
            return
        min_a, max_a = self.servo_ranges[servo_id]
        if not (min_a <= angle <= max_a):
            logging.error(f"❌ Angle {angle}° out of range ({min_a}-{max_a})")
            return
        cmd = f"MOVE {servo_id-1} {int(angle)}"
        try:
            self.serial_connection.write((cmd + "\n").encode())
            self.serial_connection.flush()
            self.current_angles[servo_id] = angle
            logging.info(f"🎯 Servo{servo_id} set to {angle}°")
        except Exception as e:
            logging.error(f"❌ Failed to send command: {e}")

    def control_loop(self):
        while self.running:
            try:
                if not self.angle_queue.empty():
                    angles = self.angle_queue.get()
                    if len(angles) == 3:
                        for i, angle in enumerate(angles, 1):
                            self.send_angle(i, angle)
            except Exception as e:
                logging.error(f"❌ Control loop error: {e}")
            time.sleep(0.05)

    def start(self):
        if not self.connected:
            self.connect()
        if not self.connected:
            logging.error("❌ Cannot start controller without connection")
            return
        self.running = True
        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        logging.info("▶️ Control loop started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.disconnect()

    def status(self):
        print("\n📊 Current Servo Angles:")
        for servo, angle in self.current_angles.items():
            print(f"Servo{servo}: {angle}°")
        print(f"Arduino: {'✅ Connected' if self.connected else '❌ Disconnected'}\n")

# ---------------- Wrapper for main.py / ik.py ----------------
def start_controller(angle_queue: multiprocessing.Queue):
    """Wrapper for starting the servo controller in a separate process."""
    port = select_serial_port(get_default_port())
    controller = RobustSerialServoController(port, angle_queue=angle_queue)
    controller.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("⏹️ Interrupted by user")
    finally:
        controller.stop()
        logging.info("👋 Controller stopped")

# ---------------- Run standalone for testing ----------------
if __name__ == "__main__":
    q = multiprocessing.Queue()
    port = select_serial_port(get_default_port())
    controller = RobustSerialServoController(port, angle_queue=q)
    controller.start()

    try:
        while True:
            cmd = input("Enter 'servo angle' or 'status' or 'exit': ").strip().lower()
            if cmd in ["exit", "quit"]:
                break
            elif cmd == "status":
                controller.status()
            else:
                try:
                    s, a = map(float, cmd.split())
                    controller.send_angle(int(s), a)
                except:
                    print("❌ Invalid input. Format: <servo_id> <angle>")
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    finally:
        controller.stop()
        print("👋 Goodbye")

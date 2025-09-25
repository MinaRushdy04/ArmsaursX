#!/usr/bin/env python3
"""
Robust Firebase Servo Control Interface
Controls Arduino servos via Firebase Realtime Database using reliable polling approach
with comprehensive error handling and cross-platform support.
"""

import serial
import time
import sys
import threading
from typing import Optional, Dict, Any
import os
import logging
import platform
import queue

try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:
    print("❌ Missing firebase_admin module. Install with `pip install firebase-admin`.")
    sys.exit(1)

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
    """Platform-agnostic serial port default."""
    if platform.system() == "Windows":
        return "COM9"
    else:
        return "/dev/ttyUSB0"


def get_serial_ports():
    """Get available serial ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def select_serial_port(default_port: str) -> str:
    """Interactive serial port selection with smart defaults."""
    available_ports = get_serial_ports()
    logging.info(f"Default port: {default_port}")

    if default_port in available_ports:
        logging.info(f"✅ {default_port} is available")
        use_default = input(f"Use {default_port}? (Y/n): ").strip().lower()
        if use_default in ['', 'y', 'yes']:
            return default_port
    else:
        logging.warning(f"⚠️  {default_port} not found in available ports.")

    if not available_ports:
        logging.error("❌ No serial ports found!")
        manual_port = input("Enter port manually: ").strip()
        return manual_port if manual_port else default_port

    print("\n📋 Available Serial Ports:")
    for i, port in enumerate(available_ports):
        print(f"{i + 1}. {port}")

    while True:
        choice = input(f"Select port number (or press Enter for {default_port}): ").strip()
        if choice == "":
            return default_port
        try:
            port_index = int(choice) - 1
            if 0 <= port_index < len(available_ports):
                return available_ports[port_index]
            else:
                print("❌ Invalid port number.")
        except ValueError:
            if choice:
                return choice
            else:
                return default_port


def check_file_exists(path: str) -> bool:
    """Check if file exists and log appropriate message."""
    if not os.path.isfile(path):
        logging.error(f"❌ Credential file not found: {path}")
        return False
    logging.info(f"✅ Found credential file: {path}")
    return True


# ---------------- Enhanced Firebase Servo Controller ----------------

class RobustFirebaseServoController:
    def __init__(self, port: str, firebase_config: dict, baudrate: int = 9600):
        """
        Initialize robust Firebase servo controller with polling approach.

        Args:
            port: Serial port for Arduino communication
            firebase_config: Firebase configuration dictionary
            baudrate: Serial communication speed (default: 9600)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.connected = False
        self.firebase_initialized = False

        if firebase_config is None:
            raise ValueError("firebase_config must be provided.")

        self.firebase_config = firebase_config
        self.servo_refs = {}

        # Servo specifications (Firebase 1-4 → Arduino 0-3)
        self.servo_ranges = {
            1: (0, 180),  # Firebase servo1 → Arduino servo0: 0° to 180°
            2: (0, 70),  # Firebase servo2 → Arduino servo1: 0° to 70°
            3: (30, 90),  # Firebase servo3 → Arduino servo2: 30° to 90°
            4: (0, 70)  # Firebase servo4 → Arduino servo3: 0° to 70°
        }

        # Track current values (Firebase servo numbers 1-4)
        self.current_values = {1: 0, 2: 0, 3: 30, 4: 0}  # Initial positions
        self.previous_values = {1: 0, 2: 0, 3: 30, 4: 0}

        # Control flags for polling thread
        self.running = False
        self.update_thread = None

        # Command queue for thread-safe operations (if needed for extensions)
        self.command_queue = queue.Queue()

        # Connection monitoring
        self.last_firebase_contact = None
        self.connection_errors = 0
        self.max_connection_errors = 5

    def firebase_to_arduino_servo(self, firebase_servo_num: int) -> int:
        """Convert Firebase servo number (1-4) to Arduino servo number (0-3)."""
        return firebase_servo_num - 1

    def arduino_to_firebase_servo(self, arduino_servo_num: int) -> int:
        """Convert Arduino servo number (0-3) to Firebase servo number (1-4)."""
        return arduino_servo_num + 1

    def initialize_firebase(self) -> bool:
        """
        Initialize Firebase connection with robust error handling.

        Returns:
            bool: True if Firebase connection successful
        """
        cred_path = self.firebase_config['credential_path']
        db_url = self.firebase_config['database_url']

        if not check_file_exists(cred_path):
            return False

        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                logging.info("🔄 Initializing Firebase Admin SDK...")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    "databaseURL": db_url
                })

            # Create references for all servos (Firebase servo1-servo4)
            for firebase_servo_num in range(1, 5):
                self.servo_refs[firebase_servo_num] = db.reference(f"servo_control/servo{firebase_servo_num}")

            self.firebase_initialized = True
            self.last_firebase_contact = time.time()
            logging.info("✅ Firebase initialized successfully")

            # Initialize Firebase values to current servo positions
            self._initialize_firebase_values()
            return True

        except Exception as e:
            logging.error(f"❌ Failed to initialize Firebase: {e}")
            return False

    def _initialize_firebase_values(self):
        """Initialize Firebase database with current servo positions."""
        logging.info("🔄 Initializing Firebase values...")
        try:
            for firebase_servo_num, angle in self.current_values.items():
                self.servo_refs[firebase_servo_num].set(angle)
                arduino_servo = self.firebase_to_arduino_servo(firebase_servo_num)
                logging.info(f"📤 Set Firebase servo{firebase_servo_num} (Arduino servo{arduino_servo}) to {angle}°")
        except Exception as e:
            logging.error(f"❌ Error initializing Firebase values: {e}")

    def connect_arduino(self) -> bool:
        """
        Connect to Arduino via serial port with robust error handling.

        Returns:
            bool: True if connection successful
        """
        try:
            logging.info(f"🔄 Connecting to Arduino on {self.port}...")
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2,
                write_timeout=2  # Add write timeout for robustness
            )
            time.sleep(2)  # Wait for Arduino to initialize
            self.connected = True
            logging.info(f"✅ Connected to Arduino on {self.port}")

            # Read any initialization messages
            self._read_responses(timeout=3)

            # Initialize Arduino servos
            self._initialize_arduino_servos()
            return True

        except serial.SerialException as e:
            logging.error(f"❌ Failed to connect to {self.port}: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ Unexpected error connecting to Arduino: {e}")
            return False

    def _initialize_arduino_servos(self):
        """Initialize Arduino servos to starting positions."""
        logging.info("🤖 Initializing Arduino servos...")
        if self.send_command("INIT"):
            logging.info("✅ Arduino servos initialized successfully")
        else:
            logging.warning("⚠️  Arduino initialization may have failed")
        time.sleep(2)

    def disconnect(self):
        """Disconnect from Arduino and stop all monitoring threads."""
        logging.info("🔄 Shutting down Firebase servo controller...")

        # Stop the update thread
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            logging.info("⏳ Waiting for update thread to stop...")
            self.update_thread.join(timeout=5)
            if self.update_thread.is_alive():
                logging.warning("⚠️  Update thread did not stop gracefully")

        # Close serial connection
        if self.serial_connection and self.connected:
            try:
                self.serial_connection.close()
                self.connected = False
                logging.info("🔌 Disconnected from Arduino")
            except Exception as e:
                logging.warning(f"⚠️  Error closing serial connection: {e}")

    def send_command(self, command: str) -> bool:
        """
        Send command to Arduino with enhanced error handling.

        Args:
            command: Command string to send

        Returns:
            bool: True if command sent and acknowledged successfully
        """
        if not self.connected or not self.serial_connection:
            logging.error("❌ Not connected to Arduino")
            return False

        try:
            # Send command
            self.serial_connection.write(f"{command}\n".encode())
            self.serial_connection.flush()  # Ensure data is sent
            logging.debug(f"📤 Sent: {command}")
            return True

        except serial.SerialTimeoutException:
            logging.error(f"❌ Serial timeout sending command: {command}")
            return False
        except Exception as e:
            logging.error(f"❌ Error sending command '{command}': {e}")
            return False

    def _read_responses(self, timeout: float = 2) -> bool:
        """
        Read and process responses from Arduino with timeout.

        Args:
            timeout: Maximum time to wait for response

        Returns:
            bool: True if valid response received
        """
        start_time = time.time()
        ack_received = False

        while (time.time() - start_time) < timeout:
            if self.serial_connection and self.serial_connection.in_waiting > 0:
                try:
                    response = self.serial_connection.readline().decode(errors='replace').strip()
                    if response:
                        logging.debug(f"📥 Arduino: {response}")
                        if "OK" in response.upper() or "READY" in response.upper():
                            ack_received = True
                except Exception as e:
                    logging.warning(f"⚠️  Error decoding Arduino response: {e}")
            time.sleep(0.01)

        return ack_received

    def _validate_servo_angle(self, firebase_servo_num: int, angle: int) -> bool:
        """
        Validate servo command parameters using Firebase servo numbers (1-4).

        Args:
            firebase_servo_num: Firebase servo number (1-4)
            angle: Target angle in degrees

        Returns:
            bool: True if parameters are valid
        """
        if firebase_servo_num not in self.servo_ranges:
            logging.error(f"❌ Invalid Firebase servo number: {firebase_servo_num}. Valid range: 1-4")
            return False

        # Type validation
        if not isinstance(angle, (int, float)):
            logging.warning(f"⚠️  Angle {angle} (type {type(angle)}) is not numeric.")
            return False

        angle = int(angle)  # Convert to int

        # Range validation
        min_angle, max_angle = self.servo_ranges[firebase_servo_num]
        if angle < min_angle or angle > max_angle:
            arduino_servo = self.firebase_to_arduino_servo(firebase_servo_num)
            logging.error(
                f"❌ Angle {angle}° out of range for Firebase servo{firebase_servo_num} "
                f"(Arduino servo{arduino_servo}). Valid range: {min_angle}°-{max_angle}°"
            )
            return False

        return True

    def _read_firebase_values(self) -> Dict[int, Any]:
        """
        Read current values from Firebase for all servos with error handling.

        Returns:
            Dict containing current Firebase values for each servo
        """
        values = {}
        try:
            for firebase_servo_num in range(1, 5):
                value = self.servo_refs[firebase_servo_num].get()
                if value is not None:
                    values[firebase_servo_num] = int(value)
                else:
                    values[firebase_servo_num] = self.current_values[firebase_servo_num]

            # Update connection status
            self.last_firebase_contact = time.time()
            self.connection_errors = 0

        except Exception as e:
            logging.error(f"❌ Error reading from Firebase: {e}")
            self.connection_errors += 1
            if self.connection_errors >= self.max_connection_errors:
                logging.error(f"❌ Too many Firebase connection errors ({self.connection_errors})")
            return self.current_values.copy()  # Return current values on error

        return values

    def _update_servo_from_firebase(self):
        """
        Main polling loop - reads from Firebase and updates servos.
        This runs in a separate thread and uses polling instead of listeners.
        """
        logging.info("🔄 Starting Firebase polling loop...")
        consecutive_errors = 0
        max_consecutive_errors = 10

        while self.running:
            try:
                # Read values from Firebase using reliable polling
                new_values = self._read_firebase_values()

                # Check for changes and update servos
                for firebase_servo_num in range(1, 5):
                    new_angle = new_values.get(firebase_servo_num, self.current_values[firebase_servo_num])

                    # Only update if value changed and is valid
                    if new_angle != self.current_values[firebase_servo_num]:
                        if self._validate_servo_angle(firebase_servo_num, new_angle):
                            # Convert Firebase servo number to Arduino servo number
                            arduino_servo_num = self.firebase_to_arduino_servo(firebase_servo_num)

                            # Send movement command (using Arduino servo number)
                            command = f"MOVE {arduino_servo_num} {new_angle}"
                            if self.send_command(command):
                                self.current_values[firebase_servo_num] = new_angle
                                logging.info(
                                    f"🎯 Firebase servo{firebase_servo_num} (Arduino servo{arduino_servo_num}) "
                                    f"updated to {new_angle}°"
                                )
                            else:
                                logging.warning(
                                    f"⚠️  Failed to update servo{firebase_servo_num} to {new_angle}°"
                                )
                        else:
                            # Reset Firebase to valid value if invalid angle received
                            try:
                                self.servo_refs[firebase_servo_num].set(self.current_values[firebase_servo_num])
                                arduino_servo = self.firebase_to_arduino_servo(firebase_servo_num)
                                logging.warning(
                                    f"🔄 Reset Firebase servo{firebase_servo_num} (Arduino servo{arduino_servo}) "
                                    f"to valid value: {self.current_values[firebase_servo_num]}°"
                                )
                            except Exception as reset_error:
                                logging.error(f"❌ Failed to reset Firebase value: {reset_error}")

                # Reset error counter on successful iteration
                consecutive_errors = 0

                # Polling interval - 10Hz update rate (100ms)
                time.sleep(0.1)

            except KeyboardInterrupt:
                logging.info("⏹️  Polling loop interrupted by user")
                break
            except Exception as e:
                consecutive_errors += 1
                logging.error(f"❌ Error in polling loop (#{consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    logging.critical(f"❌ Too many consecutive errors ({consecutive_errors}). Stopping polling loop.")
                    break

                time.sleep(1)  # Wait before retrying on error

        logging.info("⏹️  Firebase polling loop stopped")

    def start_monitoring(self) -> bool:
        """
        Start the Firebase polling thread.

        Returns:
            bool: True if monitoring started successfully
        """
        if not self.firebase_initialized:
            logging.error("❌ Firebase not initialized")
            return False

        if not self.connected:
            logging.error("❌ Arduino not connected")
            return False

        if self.running:
            logging.warning("⚠️  Monitoring already running")
            return True

        try:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_servo_from_firebase, daemon=True)
            self.update_thread.start()
            logging.info("✅ Firebase monitoring started")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to start monitoring thread: {e}")
            self.running = False
            return False

    def get_status(self):
        """Display current servo positions and system status."""
        print("\n📊 Current Servo Positions:")
        print("=" * 80)

        for firebase_servo_num in range(1, 5):
            arduino_servo_num = self.firebase_to_arduino_servo(firebase_servo_num)
            min_angle, max_angle = self.servo_ranges[firebase_servo_num]
            current = self.current_values[firebase_servo_num]
            print(
                f"Firebase servo{firebase_servo_num} (Arduino servo{arduino_servo_num}): {current}° "
                f"(range: {min_angle}°-{max_angle}°)"
            )

        # Connection status
        print("\n🔗 Connection Status:")
        print(f"Arduino: {'✅ Connected' if self.connected else '❌ Disconnected'}")
        print(f"Firebase: {'✅ Active' if self.firebase_initialized else '❌ Not initialized'}")

        if self.last_firebase_contact:
            time_since_contact = time.time() - self.last_firebase_contact
            print(f"Last Firebase contact: {time_since_contact:.1f}s ago")

        print(f"Monitoring: {'✅ Running' if self.running else '⏹️  Stopped'}")
        print("=" * 80)

    def manual_control(self, firebase_servo_num: int, angle: int) -> bool:
        """
        Manually set servo angle and update Firebase.

        Args:
            firebase_servo_num: Firebase servo number (1-4)
            angle: Target angle in degrees

        Returns:
            bool: True if successful
        """
        if not self._validate_servo_angle(firebase_servo_num, angle):
            return False

        try:
            # Update Firebase
            self.servo_refs[firebase_servo_num].set(angle)
            arduino_servo = self.firebase_to_arduino_servo(firebase_servo_num)
            logging.info(f"📤 Updated Firebase servo{firebase_servo_num} (Arduino servo{arduino_servo}) to {angle}°")
            return True
        except Exception as e:
            logging.error(f"❌ Error updating Firebase: {e}")
            return False

    def health_check(self) -> Dict[str, bool]:
        """
        Perform system health check.

        Returns:
            Dict containing health status for each component
        """
        status = {
            'arduino_connected': self.connected,
            'firebase_initialized': self.firebase_initialized,
            'monitoring_active': self.running,
            'firebase_reachable': False,
            'serial_responsive': False
        }

        # Test Firebase connectivity
        try:
            test_ref = db.reference("health_check")
            test_ref.set({"timestamp": time.time()})
            status['firebase_reachable'] = True
        except Exception as e:
            logging.warning(f"Firebase health check failed: {e}")

        # Test Arduino responsiveness
        if self.connected:
            try:
                if self.send_command("PING"):
                    status['serial_responsive'] = True
            except Exception as e:
                logging.warning(f"Arduino health check failed: {e}")

        return status


def display_servo_info(controller: RobustFirebaseServoController):
    """Display comprehensive servo specifications."""
    print("\n🤖 Servo Specifications:")
    print("=" * 80)
    for firebase_servo_num, (min_angle, max_angle) in controller.servo_ranges.items():
        arduino_servo_num = controller.firebase_to_arduino_servo(firebase_servo_num)
        print(f"Firebase servo{firebase_servo_num} (Arduino servo{arduino_servo_num}): {min_angle}° to {max_angle}°")
    print("=" * 80)


def display_usage_info():
    """Display usage information and Firebase structure."""
    print("\n🎯 Firebase Servo Control Active!")
    print("Firebase Database Structure:")
    print("  servo_control/")
    print("    ├── servo1: <angle>  → Arduino servo0 (0°-180°)")
    print("    ├── servo2: <angle>  → Arduino servo1 (0°-70°)")
    print("    ├── servo3: <angle>  → Arduino servo2 (30°-90°)")
    print("    └── servo4: <angle>  → Arduino servo3 (0°-70°)")
    print("\n📱 Update values in Firebase to control servos!")
    print("🔄 Using reliable polling method for maximum stability")
    print("Press Ctrl+C to exit\n")


def main():
    """Main application with comprehensive error handling."""
    print("🦾 Robust Firebase Arduino Servo Controller")
    print("=" * 60)

    # Firebase configuration with environment variable support
    firebase_config = {
        'credential_path': os.environ.get(
            "FIREBASE_KEY_PATH",
            "gesturefun-ab169-firebase-adminsdk-fbsvc-fbedddeb7a.json"
        ),
        'database_url': os.environ.get(
            "FIREBASE_DB_URL",
            "https://gesturefun-ab169-default-rtdb.europe-west1.firebasedatabase.app"
        )
    }

    # Interactive serial port selection
    default_port = get_default_port()
    port = select_serial_port(default_port)
    if not port:
        logging.error("❌ No port selected. Exiting...")
        return

    # Initialize controller
    controller = RobustFirebaseServoController(port, firebase_config)

    # Initialize Firebase
    if not controller.initialize_firebase():
        logging.error("❌ Failed to initialize Firebase. Please check:")
        logging.error("  • Service account key file exists and is valid")
        logging.error("  • Database URL is correct")
        logging.error("  • Internet connection is available")
        logging.error("  • Firebase project permissions are set correctly")
        return

    # Connect to Arduino
    if not controller.connect_arduino():
        logging.error("❌ Failed to connect to Arduino. Please check:")
        logging.error("  • Arduino is connected and powered")
        logging.error("  • Correct serial port selected")
        logging.error("  • Arduino code is uploaded and running")
        logging.error("  • No other programs are using the serial port")
        return

    # Display system information
    display_servo_info(controller)

    # Start Firebase monitoring
    if not controller.start_monitoring():
        logging.error("❌ Failed to start Firebase monitoring")
        controller.disconnect()
        return

    # Display usage information
    display_usage_info()

    # Main monitoring loop
    try:
        status_counter = 0
        health_check_counter = 0

        while True:
            # Show status every 5 seconds
            if status_counter >= 50:  # 50 * 0.1s = 5s
                controller.get_status()
                status_counter = 0

            # Health check every 30 seconds
            if health_check_counter >= 300:  # 300 * 0.1s = 30s
                health_status = controller.health_check()
                if not all(health_status.values()):
                    logging.warning("⚠️  System health check detected issues:")
                    for component, status in health_status.items():
                        if not status:
                            logging.warning(f"  • {component}: ❌")
                health_check_counter = 0

            time.sleep(0.1)  # 10Hz main loop
            status_counter += 1
            health_check_counter += 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Program interrupted by user")

    finally:
        controller.disconnect()
        print("👋 Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program terminated by user")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"💥 Critical error in main: {e}")
        sys.exit(1)

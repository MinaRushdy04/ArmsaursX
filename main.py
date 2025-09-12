#!/usr/bin/env python3
"""
main.py
-----------------------
Start IK.py and Arduino servo controller in parallel.
Shared queue receives joint angles from IK.py and sends them to Arduino.
"""

import multiprocessing
import time
import logging
import platform
from servo_controller import RobustSerialServoController, get_default_port, select_serial_port
from IK import run_ik, angle_queue  # Import the IK runner and shared queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


def start_controller(angle_queue: multiprocessing.Queue):
    """Start the Arduino servo controller."""
    default_port = get_default_port()
    port = select_serial_port(default_port)
    if not port:
        logging.error("❌ No port selected. Exiting...")
        return

    controller = RobustSerialServoController(port, angle_queue=angle_queue)
    controller.start_control()

    try:
        while True:
            time.sleep(0.5)  # Keep alive
    except KeyboardInterrupt:
        logging.info("⏹️  Interrupted by user")
    finally:
        controller.stop_control()
        logging.info("👋 Controller stopped")


def main():
    # Use the shared queue from IK.py
    shared_queue = angle_queue

    # Start Arduino controller
    controller_process = multiprocessing.Process(target=start_controller, args=(shared_queue,))
    controller_process.start()

    # Start IK.py loop in a separate process (manual + YOLO)
    ik_process = multiprocessing.Process(target=run_ik, args=(shared_queue,))
    ik_process.start()

    print("🦾 MOASS Controller running. Angles from IK.py will be sent to Arduino.")
    print("Type 'exit' to quit.")

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd in ["exit", "quit"]:
                break
            elif cmd:
                try:
                    servo_id, angle = map(int, cmd.split())
                    shared_queue.put((servo_id, angle))
                except ValueError:
                    print("❌ Invalid input. Format: <servo_id> <angle>")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    finally:
        controller_process.terminate()
        controller_process.join()
        ik_process.terminate()
        ik_process.join()
        print("👋 Goodbye!")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    main()
    
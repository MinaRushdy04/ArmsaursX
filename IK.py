import multiprocessing
import logging
import tkinter as tk
from servo_controller import start_controller
from IK import run_ik, angle_queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Servo angle limits and descriptions
BASE_MIN, BASE_CENTER, BASE_MAX = 0, 105, 180
SHOULDER_MIN, SHOULDER_MAX = 0, 90
ELBOW_MIN, ELBOW_MAX = 10, 90

# Coordinate offset based on reference position at angles (base: 105°, shoulder: 0°, elbow: 10°)
REFERENCE_OFFSET = (3.1232, 0, 24.7192)  # Computed via forward kinematics (L1=7, L2=8, L3=10)

def start_controller(angle_queue: multiprocessing.Queue):
    from servo_controller import RobustSerialServoController, get_default_port, select_serial_port
    default_port = get_default_port()
    port = select_serial_port(default_port)
    if not port:
        logging.error("❌ No port selected. Exiting...")
        return
    controller = RobustSerialServoController(port, angle_queue=angle_queue)
    controller.start_control()
    import time
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("⏹️ Interrupted by user")
    finally:
        controller.stop_control()
        logging.info("👋 Controller stopped")

def submit_coords():
    try:
        x = float(entry_x.get())
        y = float(entry_y.get())
        z = float(entry_z.get())
        # Apply coordinate shift: P_ik = P_real - R
        x_ik = x - REFERENCE_OFFSET[0]
        y_ik = y - REFERENCE_OFFSET[1]
        z_ik = z - REFERENCE_OFFSET[2]
        angle_queue.put(("IK", (x_ik, y_ik, z_ik)))
        result_label.config(text=f"Sent coordinates: {x_ik:.2f}, {y_ik:.2f}, {z_ik:.2f} (shifted from {x}, {y}, {z})")
    except Exception as e:
        result_label.config(text=f"❌ Invalid coordinates: {e}")

def submit_angle():
    try:
        servo_id = int(entry_servo_id.get())
        angle = float(entry_angle.get())
        if servo_id == 1 and BASE_MIN <= angle <= BASE_MAX:
            angle_queue.put((servo_id, angle))
            result_label.config(text=f"Sent: Base to {angle}° (0=right, 105=center, 180=left)")
        elif servo_id == 2 and SHOULDER_MIN <= angle <= SHOULDER_MAX:
            angle_queue.put((servo_id, angle))
            result_label.config(text=f"Sent: Shoulder to {angle}° (0=up, 90=down)")
        elif servo_id == 3 and ELBOW_MIN <= angle <= ELBOW_MAX:
            angle_queue.put((servo_id, angle))
            result_label.config(text=f"Sent: Elbow to {angle}° (90=L, 10=extended)")
        else:
            result_label.config(
                text="❌ Out of range:\nBase:0-180, Shoulder:0-90, Elbow:10-90"
            )
    except Exception as e:
        result_label.config(text=f"❌ Invalid input: {e}")

def on_close():
    root.destroy()
    controller_process.terminate()
    ik_process.terminate()

# Start processes
shared_queue = angle_queue
controller_process = multiprocessing.Process(target=start_controller, args=(shared_queue,))
controller_process.start()
ik_process = multiprocessing.Process(target=run_ik, args=(shared_queue,))
ik_process.start()

# Tkinter GUI
root = tk.Tk()
root.title("MOASS Robotic Arm Control")
root.protocol("WM_DELETE_WINDOW", on_close)

# Coordinate Input
tk.Label(root, text="Enter Coordinates (cm):").grid(row=0, column=0, columnspan=3)
tk.Label(root, text="X:").grid(row=1, column=0)
entry_x = tk.Entry(root)
entry_x.grid(row=1, column=1)
tk.Label(root, text="Y:").grid(row=2, column=0)
entry_y = tk.Entry(root)
entry_y.grid(row=2, column=1)
tk.Label(root, text="Z:").grid(row=3, column=0)
entry_z = tk.Entry(root)
entry_z.grid(row=3, column=1)
tk.Button(root, text="Submit Coords", command=submit_coords).grid(row=4, column=0, columnspan=2)

# Angle Input
tk.Label(root, text="Enter Angle:").grid(row=5, column=0, columnspan=2)
tk.Label(root, text="Servo ID (1=Base, 2=Shoulder, 3=Elbow):").grid(row=6, column=0, columnspan=2)
tk.Label(root, text="Servo ID:").grid(row=7, column=0)
entry_servo_id = tk.Entry(root)
entry_servo_id.grid(row=7, column=1)
tk.Label(root, text="Angle:").grid(row=8, column=0)
entry_angle = tk.Entry(root)
entry_angle.grid(row=8, column=1)
tk.Label(root, text="Base: 0=right, 105=center, 180=left").grid(row=9, column=0, columnspan=2)
tk.Label(root, text="Shoulder: 0=up, 90=down").grid(row=10, column=0, columnspan=2)
tk.Label(root, text="Elbow: 90=L, 10=extended").grid(row=11, column=0, columnspan=2)
tk.Button(root, text="Submit Angle", command=submit_angle).grid(row=12, column=0, columnspan=2)

# Result Label
result_label = tk.Label(root, text="Ready")
result_label.grid(row=13, column=0, columnspan=2)

root.mainloop()

# Cleanup
controller_process.join()
ik_process.join()
print("👋 Goodbye!")
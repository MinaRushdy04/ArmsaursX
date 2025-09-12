import multiprocessing
import logging
import tkinter as tk
from servo_controller import start_controller
from IK import run_ik

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def submit_coords():
    try:
        x = float(entry_x.get())
        y = float(entry_y.get())
        z = float(entry_z.get())
        shared_queue.put(("IK", (x, y, z)))  # Signal IK to process coords
        result_label.config(text=f"Sent coordinates: {x}, {y}, {z}")
    except ValueError:
        result_label.config(text="❌ Invalid coordinate input")

def submit_angle():
    try:
        servo_id = int(entry_servo_id.get())
        angle = float(entry_angle.get())
        if 1 <= servo_id <= 3 and 0 <= angle <= 180:
            shared_queue.put(("MANUAL", (servo_id, angle)))
            result_label.config(text=f"Sent angle: Servo {servo_id} to {angle}°")
        else:
            result_label.config(text="❌ Servo ID (1-3) or angle (0-180) out of range")
    except ValueError:
        result_label.config(text="❌ Invalid angle input")

def on_close():
    root.destroy()
    controller_process.terminate()
    ik_process.terminate()

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Needed for Windows

    # Shared queue for IK <-> Servo controller
    shared_queue = multiprocessing.Queue()

    # Start processes
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
    tk.Label(root, text="Servo ID (1-3):").grid(row=6, column=0)
    entry_servo_id = tk.Entry(root)
    entry_servo_id.grid(row=6, column=1)
    tk.Label(root, text="Angle (0-180°):").grid(row=7, column=0)
    entry_angle = tk.Entry(root)
    entry_angle.grid(row=7, column=1)
    tk.Button(root, text="Submit Angle", command=submit_angle).grid(row=8, column=0, columnspan=2)

    # Result Label
    result_label = tk.Label(root, text="Ready")
    result_label.grid(row=9, column=0, columnspan=2)

    root.mainloop()

    # Cleanup
    controller_process.join()
    ik_process.join()
    print("👋 Goodbye!")

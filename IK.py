# ==============================================================
#   MOASS Robotic Arm - MK2 (3DOF RRR Manipulator)
#   Forward & Inverse Kinematics Implementation
#
#   Purpose:
#   - Defines a class for the MOASS Robotic Arm (based on EEZYbotARM Mk2)
#   - Provides Forward Kinematics (FK) to compute end effector position
#   - Provides Inverse Kinematics (IK) to compute servo joint angles
#   - Receives (x, y, z) coordinates (e.g., from a camera detection system or manual input)
#   - Converts them into SG90 servo angles
#
#   Notes:
#   - Designed for small-scale, low-cost robotic arms (SG90 servos)
#   - Optimized for speed (since SG90s are slow)
#   - Visualization (optional): Uses matplotlib to plot the arm in 3D
# ==============================================================
# Developer Notes: MRR - #Last update: 09-10-2025
# Feedback to be added using FK so It can enable closed loop control and error correction
# Modifications: Added manual coordinate input and reachability warning
# ==============================================================

from math import atan2, sqrt, cos, sin, acos, degrees, radians
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Imports for Object Detection ---
from ultralytics import YOLO
import cv2
import time


class MOASS_RoboticArm:
    """
    Class representing 3DOF RRR Robotic Arm.
    Handle Inverse Kinematics.
    """

    def __init__(self):
        # Link lengths for MOASS[ArmsarousX Currently]
        self.L1 = 7.0   # Base height
        self.L2 = 8.0   # Shoulder link length
        self.L3 = 10.0   # Elbow link length

    # ----------------------------------------------------------
    # Forward Kinematics: Given angles, find end effector position
    # ----------------------------------------------------------
    def forward_kinematics(self, theta1, theta2, theta3):
        """
        Input: joint angles (in degrees)
        Output: (x, y, z) coordinates of end effector
        """

        # Convert to radians for math functions
        t1 = radians(theta1)
        t2 = radians(theta2)
        t3 = radians(theta3)

        # Planar projection length
        r = self.L2 * cos(t2) + self.L3 * cos(t2 + t3)
        z = self.L1 + self.L2 * sin(t2) + self.L3 * sin(t2 + t3)
        x = r * cos(t1)
        y = r * sin(t1)

        return (x, y, z)

    # ----------------------------------------------------------
    # Inverse Kinematics: Given target (x, y, z), find joint angles
    # ----------------------------------------------------------
    def inverse_kinematics(self, x, y, z):
        """
        Input: target position (x, y, z)
        Output: joint angles (theta1, theta2, theta3) in degrees
        """

        # Base rotation (theta1)
        theta1 = atan2(y, x)

        # Distance from base center to target (planar)
        r = sqrt(x**2 + y**2)
        dz = z - self.L1

        # Distance from shoulder joint to target point
        D = sqrt(r**2 + dz**2)

        # Check reachability
        if D > (self.L2 + self.L3) or D < abs(self.L2 - self.L3):
            raise ValueError(f"Warning: Target ({x:.1f}, {y:.1f}, {z:.1f}) is beyond arm's reach!")

        # Law of cosines for elbow angle
        cos_theta3 = (D**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)  # Numerical safety
        theta3 = acos(cos_theta3)

        # Shoulder angle (theta2)
        alpha = atan2(dz, r)
        beta = atan2(self.L3 * sin(theta3), self.L2 + self.L3 * cos(theta3))
        theta2 = alpha - beta

        return (degrees(theta1), degrees(theta2), degrees(theta3))

    # ----------------------------------------------------------
    # Optional: Plot arm in 3D for visualization
    # ----------------------------------------------------------
    def plot_arm(self, theta1, theta2, theta3):
        """
        Visualize the arm in 3D using matplotlib.
        """

        # Forward kinematics for each joint
        t1 = radians(theta1)
        t2 = radians(theta2)
        t3 = radians(theta3)

        # Base
        x0, y0, z0 = 0, 0, 0
        # Shoulder
        x1, y1, z1 = 0, 0, self.L1
        # Elbow
        x2 = x1 + self.L2 * cos(t2) * cos(t1)
        y2 = y1 + self.L2 * cos(t2) * sin(t1)
        z2 = z1 + self.L2 * sin(t2)
        # End effector
        x3 = x2 + self.L3 * cos(t2 + t3) * cos(t1)
        y3 = y2 + self.L3 * cos(t2 + t3) * sin(t1)
        z3 = z2 + self.L3 * sin(t2 + t3)

        # Plot points
        xs = [x0, x1, x2, x3]
        ys = [y0, y1, y2, y3]
        zs = [z0, z1, z2, z3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=6)

        # Labels
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title("MOASS Robotic Arm")

        plt.show()


# ----------------------------------------------------------
# Main execution loop for Object Detection and IK Calculation
# ----------------------------------------------------------
if __name__ == "__main__":
    arm = MOASS_RoboticArm()

    # --- Object Detection Setup ---
    # IMPORTANT: Replace "/home/sandy/yolo_model/best.pt" with the actual path to your YOLO model
    try:
        model = YOLO("/home/sandy/yolo_model/best.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the model path is correct and ultralytics is installed.")
        exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- Camera to Robot Base Frame Conversion Parameters ---
    CM_TO_PIXEL = 32.0 / 640  # Example: 32cm field of view for 640 pixels
    BASE_OFFSET_X = 5.0       # Example: Offset in X from camera center to robot base
    BASE_OFFSET_Y = 10.0      # Example: Offset in Y from camera center to robot base
    OBJECT_HEIGHT = 0.0       # Example: Assumed Z-height of the object (e.g., on the table)

    # --- Input Mode Selection ---
    print("Choose input mode:")
    print("  'm' for manual coordinates (until camera is calibrated)")
    print("  'c' for camera (YOLO detection)")
    mode = input("Enter 'm' or 'c': ").strip().lower()
    use_manual_input = mode == 'm'

    prev_time = time.time()

    print("Starting object detection and IK calculation...")
    print("Press 'q' to quit.")

    while True:
        detected_object_coords = None

        if use_manual_input:
            # Manual coordinate input
            try:
                coords_input = input("Enter x, y, z (in cm, comma-separated, e.g., 10,0,7): ").strip()
                x, y, z = map(float, coords_input.split(','))
                detected_object_coords = (x, y, z)
                print(f"Manual target: {detected_object_coords}")
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter three numbers separated by commas (e.g., 10,0,7).")
                continue
        else:
            # Camera/YOLO detection
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            results = model(frame, verbose=False)

            for r in results:
                if r.boxes:
                    box = r.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    x_cm_camera = cx * CM_TO_PIXEL
                    y_cm_camera = cy * CM_TO_PIXEL
                    X_base = x_cm_camera - BASE_OFFSET_X
                    Y_base = y_cm_camera - BASE_OFFSET_Y
                    Z_base = OBJECT_HEIGHT
                    detected_object_coords = (X_base, Y_base, Z_base)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.putText(frame, f"X={X_base:.1f}cm, Y={Y_base:.1f}cm, Z={Z_base:.1f}cm", (int(x1), int(y2)+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                    break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.imshow("YOLO Detection and IK", frame)

        # --- IK Calculation and Output ---
        if detected_object_coords:
            try:
                joint_angles = arm.inverse_kinematics(*detected_object_coords)
                print(f"Target: {detected_object_coords} -> Joint Angles: {joint_angles}")
                # --- Here you would send joint_angles to Arduino via serial ---
            except ValueError as e:
                if "beyond arm's reach" in str(e):
                    print(e)  # Print reachability warning
                else:
                    print(f"Error calculating Inverse Kinematics for {detected_object_coords}: {e}")
                    print("Ensure the target is within the arm's reachable workspace and IK equations are valid.")
            except Exception as e:
                print(f"Unexpected error in IK for {detected_object_coords}: {e}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

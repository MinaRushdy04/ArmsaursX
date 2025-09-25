#!/usr/bin/env python3
"""
Advanced Robotic Arm Controller with Computer Vision, Manual Control, and Sorting
================================================================================

This application combines YOLO object detection with inverse kinematics to control
a robotic arm for autonomous object grabbing and color-based sorting, while also 
providing manual servo control through an intuitive GUI interface.

Author: Robotic Arm Control System
Version: 2.1 - Added Sorting Functionality
Python: 3.8+
License: MIT

NEW FEATURES in v2.1:
-------------------
• Color-based object sorting functionality
• Automatic yellow object sorting (drop at base 180°)
• Automatic red object sorting (drop at base 0°)
• Manual sorting trigger button
• Enhanced sorting sequence with pick-and-drop logic

Features:
---------
• Real-time object detection and tracking using YOLO
• Automatic inverse kinematics calculation for target positions  
• Multi-object selection and autonomous grabbing
• Color-based sorting with configurable drop zones
• Manual servo control with real-time feedback
• Camera calibration and coordinate transformation
• Thread-safe GUI with tabbed interface
• Serial communication with Arduino
• Safety features and emergency controls

Hardware Requirements:
--------------------
• Arduino Uno/Nano with servo control firmware
• 4 servo motors (Base, Shoulder, Elbow, Gripper)
• USB webcam for object detection
• Armosours or similar robotic arm configuration

Software Dependencies:
--------------------
pip install ultralytics opencv-python numpy pyserial

Usage:
------
1. Connect Arduino and camera to computer
2. Place your trained YOLO model as 'best.pt' in the same directory
3. Run: python enhanced_robotic_arm_controller.py
4. Configure connections in the Connection tab
5. Use Vision Control tab for autonomous operation and sorting
6. Use Manual Control tab for direct servo control

Architecture:
------------
• DetectedObject: Data class for object information
• CameraCalibration: Handles pixel-to-world coordinate transformation
• InverseKinematics: Calculates servo angles for target positions
• SerialController: Manages Arduino communication with sorting capabilities
• VisionSystem: Processes camera frames and detects objects
• RoboticArmGUI: Main application interface with tabs and sorting controls

Safety Features:
---------------
• Servo angle limits enforcement
• Emergency stop functionality
• Connection status monitoring
• Reachability analysis before attempting grabs
• Thread cleanup on application exit
• Safe drop zone validation
"""

import logging
import time
import threading
import queue
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import math
import serial
import serial.tools.list_ports

# Handle YOLO import with fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Using mock YOLO for testing.")

# Configure logging system
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('robotic_arm.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """
    Container for detected object information

    Attributes:
        id: Unique identifier for tracking across frames
        name: Object class name from YOLO model
        confidence: Detection confidence score (0.0-1.0)
        center_pixel: (x, y) pixel coordinates of object center
        world_coords: (x, y, z) world coordinates in millimeters
        servo_angles: (base, shoulder, elbow) angles in degrees, None if unreachable
        bbox: (x1, y1, x2, y2) bounding box coordinates
        reachable: Boolean indicating if object is within arm workspace
    """
    id: int
    name: str
    confidence: float
    center_pixel: Tuple[int, int]
    world_coords: Tuple[float, float, float]
    servo_angles: Optional[Tuple[float, float, float]]
    bbox: Tuple[int, int, int, int]
    reachable: bool = True


class CameraCalibration:
    """
    Camera calibration parameters and coordinate transformation system

    Handles conversion from 2D pixel coordinates to 3D world coordinates
    using camera intrinsic/extrinsic parameters obtained from calibration.
    """

    def __init__(self):
        """Initialize camera calibration parameters"""
        # Camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        self.camera_matrix = np.array([
            [546.25061067, 0., 317.92410558],
            [0., 548.85987516, 239.71830120],
            [0., 0., 1.]
        ])

        # Distortion coefficients [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([
            [0.23208356, -1.03167154, -0.00369298, -0.00400159, 2.99040251]
        ])

        # Rotation matrix (camera to world)
        self.R_cam = np.array([
            [0.99418228, 0.03701466, -0.10115093],
            [-0.03531900, 0.99920477, 0.01850404],
            [0.10175541, -0.01482384, 0.99469899]
        ])

        # Translation vector (camera position in world coordinates)
        self.t_cam = np.array([[-1.98635758], [-1.26039358], [6.79782214]])

        # Coordinate system adjustments
        self.positional_offset = (-5.5, 13)  # (x_offset, y_offset) in cm
        self.scale_factor = 2.4  # Scaling factor for coordinate conversion

    def pixel_to_world(self, u_px: int, v_px: int) -> Tuple[float, float, float, Tuple[float, float]]:
        """
        Convert pixel coordinates to world coordinates

        Args:
            u_px: Pixel x-coordinate
            v_px: Pixel y-coordinate

        Returns:
            Tuple containing:
            - X: World X coordinate in mm
            - Y: World Y coordinate in mm  
            - s: Scale factor from projection
            - (u_corr, v_corr): Undistorted pixel coordinates
        """
        # Undistort pixel coordinates
        pts = np.array([[[u_px, v_px]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        u_corr, v_corr = und[0, 0]

        # Project to world plane (z=0)
        p = np.array([u_corr, v_corr, 1.0]).reshape(3, 1)
        M = self.camera_matrix @ self.R_cam[:, :2]  # First 2 columns for z=0 plane
        b = self.camera_matrix @ self.t_cam

        # Solve for world coordinates
        A = np.hstack((M, -p))
        rhs = -b

        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Use least squares if matrix is singular
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        # Apply scaling and offsets
        X = float(sol[0]) * self.scale_factor + self.positional_offset[0]
        Y = float(sol[1]) * self.scale_factor + self.positional_offset[1]
        s = float(sol[2])

        return X * 10, Y * 10, s, (u_corr, v_corr)  # Convert cm to mm


class InverseKinematics:
    """
    Inverse kinematics calculator for robotic arm

    Calculates required servo angles to reach target positions in 3D space
    using geometric approach with law of cosines.
    """

    def __init__(self):
        """Initialize arm physical parameters"""
        # Physical dimensions in millimeters
        self.L1 = 135.0  # Upper arm segment length
        self.L2 = 147.0  # Forearm segment length  
        self.BASE_HEIGHT = 65.0  # Height of shoulder joint above base

    def calculate(self, x: float, y: float, z: float) -> Optional[Tuple[float, float, float]]:
        """
        Calculate servo angles for target position using inverse kinematics

        Process:
        1. Calculate base rotation angle using atan2(y,x)
        2. Find planar distance from shoulder to target
        3. Use law of cosines to find elbow and shoulder angles
        4. Apply servo-specific corrections for physical mounting

        Args:
            x: Target X coordinate in mm
            y: Target Y coordinate in mm
            z: Target Z coordinate in mm

        Returns:
            Tuple of (base_angle, shoulder_angle, elbow_angle) in degrees
            Returns None if position is unreachable
        """
        try:
            # Adjust Z coordinate relative to shoulder joint
            z_adjusted = z - self.BASE_HEIGHT

            # Calculate base servo angle (horizontal rotation)
            base_angle = math.degrees(math.atan2(y, x))
            if base_angle < 0:
                base_angle += 360

            # Calculate distances
            horizontal_distance = math.hypot(x, y)
            target_distance = math.hypot(horizontal_distance, z_adjusted)

            # Check if target is within workspace
            max_reach = self.L1 + self.L2  # Fully extended
            min_reach = abs(self.L1 - self.L2)  # Fully folded

            if target_distance > max_reach or target_distance < min_reach or target_distance == 0:
                return None  # Position unreachable

            # Calculate elbow angle using law of cosines
            cos_elbow_internal = (self.L1**2 + self.L2**2 - target_distance**2) / (2 * self.L1 * self.L2)
            cos_elbow_internal = max(-1.0, min(1.0, cos_elbow_internal))  # Clamp to valid range
            elbow_internal_angle = math.degrees(math.acos(cos_elbow_internal))
            elbow_servo_angle = 180.0 - elbow_internal_angle

            # Calculate shoulder angle using law of cosines
            cos_shoulder_offset = (self.L1**2 + target_distance**2 - self.L2**2) / (2 * self.L1 * target_distance)
            cos_shoulder_offset = max(-1.0, min(1.0, cos_shoulder_offset))
            shoulder_offset_angle = math.degrees(math.acos(cos_shoulder_offset))

            # Calculate elevation angle
            elevation_angle = math.degrees(math.atan2(z_adjusted, horizontal_distance))
            shoulder_angle_raw = elevation_angle + shoulder_offset_angle

            # Apply servo mounting corrections
            if base_angle > 180:
                base_angle = base_angle - 360

            # Convert to servo angles (specific to hardware setup)
            base_servo = 196 - base_angle
            shoulder_servo = 90 - shoulder_angle_raw
            elbow_servo = elbow_servo_angle - 90

            return base_servo, shoulder_servo, elbow_servo

        except Exception as e:
            logger.error(f"IK calculation error: {e}")
            return None


class SerialController:
    """
    Arduino serial communication handler with sorting capabilities

    Manages connection and command sending to Arduino for servo control.
    Includes safety features like angle limits and command validation.
    Enhanced with color-based sorting functionality.
    """

    def __init__(self, port: str = "COM9", baudrate: int = 9600):
        """
        Initialize serial controller

        Args:
            port: Serial port name (e.g., 'COM9', '/dev/ttyUSB0')
            baudrate: Communication speed in bps
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.connected = False

        # Servo angle limits (Arduino servo numbers 0-3)
        self.servo_limits = {
            0: (0, 180),    # Base servo: 0° to 180°
            1: (0, 70),     # Shoulder servo: 0° to 70°
            2: (10, 90),    # Elbow servo: 10° to 90°
            3: (0, 70)      # Gripper servo: 0° to 70°
        }

        # Sorting configuration
        self.yellow_drop_zone_big = 180  # Base angle for yellow objects
        self.red_drop_zone_big = 15       # Base angle for red objects
        # med
        self.yellow_drop_zone_med = 150  # Base angle for yellow objects
        self.red_drop_zone_med = 50       # Base angle for red objects

        # Drop zone positions (fixed shoulder and elbow for dropping)
        self.drop_shoulder_angle = 5  # Safe drop shoulder position
        self.drop_elbow_angle = 10     # Safe drop elbow position

        # Drop to a big-sized cube to box based
        self.extend_shoulder_angle_big = 40
        self.extend_elbow_angle_big = 30
        self.griper_extend_big = 15

        # Drop a medium-sized cube to the box
        self.extend_shoulder_angle_medium = 50
        self.extend_elbow_angle_medium = 70
        self.griper_extend_med = 3



    def connect(self, port: str = None) -> bool:
        """
        Establish serial connection to Arduino

        Args:
            port: Optional port override

        Returns:
            True if connection successful, False otherwise
        """
        if port:
            self.port = port

        try:
            self.serial_connection = serial.Serial(
                port=self.port, 
                baudrate=self.baudrate, 
                timeout=2
            )
            time.sleep(2)  # Allow Arduino to initialize
            self.connected = True
            logger.info(f"Successfully connected to {self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Close serial connection and cleanup"""
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except:
                pass
        self.connected = False
        logger.info("Disconnected from Arduino")

    def send_command(self, command: str) -> bool:
        """
        Send command string to Arduino

        Args:
            command: Command string (e.g., "MOVE 0 90", "INIT")

        Returns:
            True if command sent successfully, False otherwise
        """
        if not (self.connected and self.serial_connection):
            logger.warning("Cannot send command: not connected to Arduino")
            return False

        try:
            self.serial_connection.write(f"{command}\n".encode("ascii", errors="ignore"))
            self.serial_connection.flush()
            logger.debug(f"Sent command: {command}")
            return True
        except Exception as e:
            logger.error(f"Send command error: {e}")
            return False

    def move_servo(self, servo_num: int, angle: float) -> bool:
        """
        Move single servo to specified angle with safety checks

        Args:
            servo_num: Servo number (0-3)
            angle: Target angle in degrees

        Returns:
            True if movement command sent successfully
        """
        if servo_num not in self.servo_limits:
            logger.error(f"Invalid servo number: {servo_num}")
            return False

        # Apply angle limits
        min_angle, max_angle = self.servo_limits[servo_num]
        angle = max(min_angle, min(max_angle, int(angle)))

        return self.send_command(f"MOVE {servo_num} {angle}")

    def grab_sequence(self, base: float, shoulder: float, elbow: float):
        """
        Execute complete grab sequence for detected object

        Sequence:
        1. Initialize to safe position
        2. Open gripper
        3. Move to target position (base, shoulder, elbow)
        4. Close gripper
        5. Return to safe position

        Args:
            base: Base servo angle in degrees
            shoulder: Shoulder servo angle in degrees
            elbow: Elbow servo angle in degrees
        """
        logger.info(f"Executing grab sequence: Base={base:.1f}°, Shoulder={shoulder:.1f}°, Elbow={elbow:.1f}°")

        # Define grab sequence with timing
        commands = [
            ("INIT", 1.0),              # Initialize to safe position
            (f"MOVE 3 60", 0.3),        # Open gripper
            (f"MOVE 0 {int(base)}", 0.3),     # Move base
            (f"MOVE 1 {int(shoulder)}", 0.3), # Move shoulder
            (f"MOVE 2 {int(elbow)}", 0.3),    # Move elbow
            (f"MOVE 3 7", 1.0),         # Close gripper (grab object)
            ("INIT", 1.0)               # Return to safe home position
        ]

        # Execute sequence with error handling
        for command, delay in commands:
            if self.send_command(command):
                time.sleep(delay)
            else:
                logger.error(f"Grab sequence failed at command: {command}")
                # Emergency return to safe position
                self.send_command("INIT")
                break

        logger.info("Grab sequence completed")

    def sorting_sequence(self, pickup_base: float, pickup_shoulder: float, pickup_elbow: float, drop_base: float, type ="big"):
        """
        Execute complete sorting sequence: pick up object and drop at specified location

        Sequence:
        1. Initialize to safe position
        2. Open gripper
        3. Move to pickup position
        4. Close gripper (grab object)
        5. Lift object safely
        6. Move to drop zone
        7. Open gripper (drop object)
        8. Return to safe position

        Args:
            pickup_base: Base angle for pickup position
            pickup_shoulder: Shoulder angle for pickup position
            pickup_elbow: Elbow angle for pickup position
            drop_base: Base angle for drop zone
        """
        elbow_extend = 0
        shoulder_extend = 0
        gripper_extend = 0
        if type == "big":
            elbow_extend = self.extend_elbow_angle_big
            shoulder_extend = self.extend_shoulder_angle_big
            gripper_extend = self.griper_extend_big
        else:
            elbow_extend = self.extend_elbow_angle_medium
            shoulder_extend = self.extend_shoulder_angle_medium
            gripper_extend = self.griper_extend_med


        logger.info(f"Executing sorting sequence: Pickup({pickup_base:.1f}°, {pickup_shoulder:.1f}°, {pickup_elbow:.1f}°) -> Drop({drop_base:.1f}°)")

        # Define sorting sequence with timing
        commands = [
            ("INIT", 1.0),                                    # Initialize to safe position
            (f"MOVE 3 60", 0.3),                             # Open gripper wide
            (f"MOVE 0 {int(pickup_base)}", 0.4),             # Move base to pickup position
            (f"MOVE 1 {int(pickup_shoulder)}", 0.4),         # Move shoulder to pickup position
            (f"MOVE 2 {int(pickup_elbow)}", 0.4),            # Move elbow to pickup position
            (f"MOVE 3 {gripper_extend}", 0.8),                             # Close gripper (grab object)
            (f"MOVE 2 {self.drop_elbow_angle}", 0.3),        # Lift elbow to safe transport position
            (f"MOVE 1 {self.drop_shoulder_angle}", 0.3),     # Move shoulder to safe transport position
            (f"MOVE 0 {int(drop_base)}", 0.8),               # Move base to drop zone
            (f"MOVE 2 {elbow_extend}", 0.3),     # Ensure shoulder is at drop position
            (f"MOVE 1 {shoulder_extend}", 0.3),        # Ensure elbow is at drop position
            (f"MOVE 3 60", 0.5),                            # Open gripper (drop object)
            ("INIT", 1.0)                                    # Return to safe home position
        ]

        # Execute sequence with error handling
        for command, delay in commands:
            if self.send_command(command):
                time.sleep(delay)
            else:
                logger.error(f"Sorting sequence failed at command: {command}")
                # Emergency return to safe position
                self.send_command("INIT")
                break

        logger.info(f"Sorting sequence completed - object dropped at base {drop_base}°")

    def sort_objects(self, detected_objects: List):
        """
        Automatic sorting function for detected objects

        Processes list of detected objects and sorts them based on color

        Args:
            detected_objects: List of DetectedObject instances
        """
        if not self.connected:
            logger.warning("Cannot sort objects: not connected to Arduino")
            return {"yellow": 0, "red": 0, "other": 0}

        sorted_count = {"yellow": 0, "red": 0, "other": 0}

        for obj in detected_objects:
            if not obj.reachable or not obj.servo_angles or obj.confidence < 0.5:
                continue

            object_name_lower = obj.name.lower()
            base, shoulder, elbow = obj.servo_angles
            delay_=6
            type = ""
            if "big" in object_name_lower:
                type = "big"
            else:
                type = "medium"

            # Check for yellow objects
            if "yellow" in object_name_lower:
                logger.info(f"Sorting YELLOW object: {obj.name} (ID: {obj.id})")
                if "big" in object_name_lower:
                    drop_zone = self.yellow_drop_zone_big
                else:
                    drop_zone = self.yellow_drop_zone_med
                self.sorting_sequence(base, shoulder, elbow, drop_zone,type)
                sorted_count["yellow"] += 1
                time.sleep(delay_)  # Brief pause between sorts

            # Check for red objects  
            elif "red" in object_name_lower:
                logger.info(f"Sorting RED object: {obj.name} (ID: {obj.id})")
                if "big" in object_name_lower:
                    drop_zone = self.red_drop_zone_big
                else:
                    drop_zone = self.red_drop_zone_med
                self.sorting_sequence(base, shoulder, elbow, drop_zone,type)
                sorted_count["red"] += 1
                time.sleep(delay_)  # Brief pause between sorts

            else:
                logger.info(f"Object {obj.name} does not match sorting criteria (not red or yellow)")
                sorted_count["other"] += 1

        # Log sorting summary
        total_sorted = sorted_count["yellow"] + sorted_count["red"]
        logger.info(f"Sorting completed: {sorted_count['yellow']} yellow, {sorted_count['red']} red, {sorted_count['other']} other objects. Total sorted: {total_sorted}")

        return sorted_count


# Mock YOLO class for testing when ultralytics is not available
class MockYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {0: "yellow_object", 1: "red_object", 2: "other_object"}
        logger.info(f"Mock YOLO initialized (ultralytics not available)")

    def __call__(self, frame, verbose=False):
        # Return mock results for testing
        return [MockResult()]

class MockResult:
    def __init__(self):
        self.boxes = None  # No detections in mock mode


class VisionSystem:
    """
    Computer vision and object detection system

    Handles camera initialization, frame processing, object detection using YOLO,
    and coordinate transformation from pixel to world coordinates.
    """

    def __init__(self, model_path: str = "./best.pt", camera_index: int = 1):
        """
        Initialize vision system

        Args:
            model_path: Path to YOLO model file
            camera_index: Camera device index (usually 0 or 1)
        """
        try:
            if YOLO_AVAILABLE:
                self.model = YOLO(model_path)
                logger.info(f"YOLO model loaded from {model_path}")
            else:
                self.model = MockYOLO(model_path)
                logger.warning("Using mock YOLO - install ultralytics for real detection")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.warning("Using mock YOLO instead")
            self.model = MockYOLO(model_path)

        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None

        # Initialize subsystems
        self.calibration = CameraCalibration()
        self.kinematics = InverseKinematics()

        # Object tracking
        self.object_id_counter = 0
        self.tracked_objects: Dict[int, Tuple[int, int]] = {}
        self.distance_threshold = 50  # Pixels

        # System state
        self.running = False

    def initialize_camera(self) -> bool:
        """
        Initialize camera capture

        Returns:
            True if camera initialized successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info(f"Camera {self.camera_index} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def process_frame(self) -> Tuple[Optional[np.ndarray], List[DetectedObject]]:
        """
        Process single camera frame for object detection

        Returns:
            Tuple of (annotated_frame, detected_objects_list)
            Returns (None, []) if frame capture fails
        """
        if not self.cap:
            return None, []

        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            return None, []

        detected_objects = []
        im_h, im_w = frame.shape[:2]

        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    # Extract detection data
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Skip low confidence detections
                    if conf < 0.3:
                        continue

                    # Calculate center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Track object across frames
                    object_id = self._track_object(cx, cy)

                    # Convert to world coordinates
                    Xw, Yw, scale_s, (u_corr, v_corr) = self.calibration.pixel_to_world(cx, cy)

                    # Calculate servo angles using inverse kinematics
                    z_world = 10.0  # Assume objects are on table surface
                    servo_angles = self.kinematics.calculate(Xw, Yw, z_world)
                    reachable = servo_angles is not None

                    # Create detected object instance
                    detected_obj = DetectedObject(
                        id=object_id,
                        name=self.model.names[cls],
                        confidence=conf,
                        center_pixel=(cx, cy),
                        world_coords=(Xw, Yw, z_world),
                        servo_angles=servo_angles,
                        bbox=(x1, y1, x2, y2),
                        reachable=reachable
                    )

                    detected_objects.append(detected_obj)

                    # Draw visual annotations with sorting info
                    self._draw_annotations(frame, detected_obj, im_w, im_h)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

        return frame, detected_objects

    def _draw_annotations(self, frame: np.ndarray, obj: DetectedObject, im_w: int, im_h: int):
        """
        Draw visual annotations on frame for detected object
        Enhanced with sorting color indicators

        Args:
            frame: Image frame to annotate
            obj: DetectedObject to visualize
            im_w: Image width
            im_h: Image height
        """
        x1, y1, x2, y2 = obj.bbox
        cx, cy = obj.center_pixel

        # Determine object color and sorting info
        object_name_lower = obj.name.lower()
        is_yellow = "yellow" in object_name_lower
        is_red = "red" in object_name_lower
        is_sortable = is_yellow or is_red

        # Choose color based on reachability and sorting capability
        if not obj.reachable:
            color = (0, 0, 255)  # Red for unreachable
            sort_info = "UNREACHABLE"
        elif is_yellow:
            color = (0, 255, 255)  # Yellow for yellow objects
            sort_info = "SORT→180°"
        elif is_red:
            color = (0, 0, 255)  # Red for red objects  
            sort_info = "SORT→0°"
        elif obj.reachable:
            color = (0, 255, 0)  # Green for reachable but not sortable
            sort_info = "NO SORT"
        else:
            color = (128, 128, 128)  # Gray for other
            sort_info = "UNKNOWN"

        # Draw bounding box with thicker line for sortable objects
        line_thickness = 3 if is_sortable else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

        # Draw center point and crosshairs
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (0, cy), (im_w, cy), (0, 0, 255), 1)
        cv2.line(frame, (cx, 0), (cx, im_h), (0, 0, 255), 1)

        # Text annotations with enhanced sorting info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Object info with sorting indicator
        sort_indicator = "🟡" if is_yellow else "🔴" if is_red else "⚫"
        cv2.putText(frame, f"ID:{obj.id} {obj.name} {obj.confidence:.2f}", 
                   (x1, y1-60), font, font_scale, color, thickness)

        # World coordinates
        cv2.putText(frame, f"World: {obj.world_coords[0]:.1f}, {obj.world_coords[1]:.1f}", 
                   (x1, y1-45), font, font_scale, (0, 255, 255), thickness)

        # Servo angles if reachable
        if obj.servo_angles:
            angles_text = f"Angles: {obj.servo_angles[0]:.1f}, {obj.servo_angles[1]:.1f}, {obj.servo_angles[2]:.1f}"
            cv2.putText(frame, angles_text, 
                       (x1, y1-30), font, 0.4, (200, 200, 200), thickness)

        # Draw sorting zone indicator if sortable
        if is_sortable and obj.reachable:
            zone_color = (0, 255, 255) if is_yellow else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 15, zone_color, 2)

    def _track_object(self, cx: int, cy: int) -> int:
        """
        Track objects across frames using position-based matching

        Args:
            cx: Object center x-coordinate
            cy: Object center y-coordinate

        Returns:
            Object ID (existing or new)
        """
        # Find existing object within distance threshold
        for obj_id, (px, py) in self.tracked_objects.items():
            if abs(cx - px) < self.distance_threshold and abs(cy - py) < self.distance_threshold:
                # Update position
                self.tracked_objects[obj_id] = (cx, cy)
                return obj_id

        # Create new object ID
        self.object_id_counter += 1
        self.tracked_objects[self.object_id_counter] = (cx, cy)
        return self.object_id_counter

    def cleanup(self):
        """Release camera resources and cleanup"""
        if self.cap:
            self.cap.release()
        logger.info("Vision system cleaned up")


class RoboticArmGUI:
    """
    Main GUI application combining vision, manual control, and sorting functionality

    Provides tabbed interface for:
    - Vision Control: Object detection, autonomous grabbing, and sorting
    - Manual Control: Direct servo position control
    - Connection: Hardware setup and testing
    """

    def __init__(self):
        """Initialize GUI application"""
        logger.info("Initializing GUI application...")

        # Create main window
        self.root = tk.Tk()
        self.root.title("🦾 Advanced Robotic Arm Controller with Sorting")
        self.root.geometry("1200x900")  # Increased height for sorting controls
        self.root.resizable(True, True)

        # Initialize subsystems
        try:
            self.serial_controller = SerialController()
            self.vision_system = VisionSystem()
            logger.info("Subsystems initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize subsystems: {e}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize components:\n\n{str(e)}\n\nThe application will continue but some features may not work.")
            self.serial_controller = SerialController()
            self.vision_system = None

        # GUI state variables
        self.vision_running = tk.BooleanVar(master=self.root, value=False)
        self.auto_grab_enabled = tk.BooleanVar(master=self.root, value=False)
        self.auto_sort_enabled = tk.BooleanVar(master=self.root, value=False)
        self.selected_objects = set()  # Set of object names to grab

        # Threading components
        self.vision_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.objects_queue = queue.Queue(maxsize=10)

        # Widget references
        self.servo_sliders = {}
        self.servo_labels = {}
        self.servo_entries = {}
        self.object_listbox = None
        self.status_label = None
        self.connect_button = None
        self.vision_button = None
        self.sort_button = None  # NEW: Sort button reference
        self.sorting_status = None

        # Initialize GUI
        try:
            self.setup_gui()
            self.update_display()
            logger.info("GUI initialized successfully with sorting functionality")
        except Exception as e:
            logger.error(f"GUI setup failed: {e}")
            messagebox.showerror("GUI Error", f"Failed to setup GUI:\n\n{str(e)}")
            raise

    def setup_gui(self):
        """Setup the complete GUI interface with tabs"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create tabbed notebook
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        vision_frame = ttk.Frame(notebook, padding="10")
        manual_frame = ttk.Frame(notebook, padding="10")
        connection_frame = ttk.Frame(notebook, padding="10")

        notebook.add(vision_frame, text="🔍 Vision & Sorting")  # Updated tab name
        notebook.add(manual_frame, text="🎮 Manual Control")
        notebook.add(connection_frame, text="🔌 Connection")

        # Setup tab contents
        self.setup_vision_tab(vision_frame)
        self.setup_manual_tab(manual_frame)
        self.setup_connection_tab(connection_frame)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready - Sorting system loaded", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def setup_vision_tab(self, parent):
        """Setup vision control interface with sorting functionality"""
        # Vision control section
        control_frame = ttk.LabelFrame(parent, text="Vision Control", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # First row of controls
        control_row1 = ttk.Frame(control_frame)
        control_row1.pack(fill=tk.X, pady=(0, 10))

        self.vision_button = ttk.Button(
            control_row1, 
            text="Start Vision", 
            command=self.toggle_vision,
            width=15
        )
        self.vision_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Checkbutton(
            control_row1, 
            text="Auto Grab Enabled", 
            variable=self.auto_grab_enabled
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Checkbutton(
            control_row1, 
            text="Auto Sort Enabled", 
            variable=self.auto_sort_enabled
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            control_row1, 
            text="Clear Selections", 
            command=self.clear_object_selections,
            width=15
        ).pack(side=tk.LEFT)

        # NEW: Sorting control section
        sorting_frame = ttk.LabelFrame(parent, text="🎯 Sorting Controls", padding="10")
        sorting_frame.pack(fill=tk.X, pady=(0, 10))

        # Sorting instructions
        # sort_instructions = ttk.Label(
        #     sorting_frame,
        #     text="Sorting Rules: BIG Yellow objects → Base 150° |  BIG Red objects → Base 15° | Other objects ignored",
        #     font=("Arial", 9, "italic"),
        #     foreground="blue"
        # )
        # sort_instructions.pack(anchor=tk.W, pady=(0, 10))

        # Sorting buttons
        sort_button_frame = ttk.Frame(sorting_frame)
        sort_button_frame.pack(fill=tk.X)

        self.sort_button = ttk.Button(
            sort_button_frame,
            text="🎯 Start Sorting Now",
            command=self.manual_sort_objects,
            width=20
        )
        self.sort_button.pack(side=tk.LEFT, padx=(0, 15))

        # Drop zone configuration display
        config_frame = ttk.Frame(sort_button_frame)
        config_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(config_frame, text="Drop Zones:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(config_frame, text="🟡 BIG Yellow → 180°", font=("Arial", 8), foreground="orange").pack(anchor=tk.W)
        ttk.Label(config_frame, text="🔴 BIG Red → 0°", font=("Arial", 8), foreground="red").pack(anchor=tk.W)

        # Emergency stop (moved to more prominent position)
        ttk.Button(
            sort_button_frame, 
            text="🛑 Emergency Stop", 
            command=self.emergency_stop,
            width=15
        ).pack(side=tk.RIGHT)

        # Object selection section
        selection_frame = ttk.LabelFrame(parent, text="Object Selection & Detection", padding="10")
        selection_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions label
        instructions = ttk.Label(
            selection_frame, 
            text="Detected objects appear below. Select for manual grabbing or enable auto-sort for automatic processing:",
            font=("Arial", 10)
        )
        instructions.pack(anchor=tk.W, pady=(0, 10))

        # Listbox with scrollbar
        list_frame = ttk.Frame(selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.object_listbox = tk.Listbox(
            list_frame, 
            selectmode=tk.MULTIPLE, 
            height=10,
            font=("Arial", 10)
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.object_listbox.yview)
        self.object_listbox.configure(yscrollcommand=scrollbar.set)

        self.object_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.object_listbox.bind('<<ListboxSelect>>', self.on_object_select)

        # Action buttons
        action_frame = ttk.Frame(selection_frame)
        action_frame.pack(fill=tk.X, pady=(15, 0))

        ttk.Button(
            action_frame, 
            text="Grab Selected Now", 
            command=self.grab_selected_objects,
            width=20
        ).pack(side=tk.LEFT, padx=(0, 10))

        # Status display for sorting
        self.sorting_status = ttk.Label(
            action_frame,
            text="Sorting Status: Ready",
            font=("Arial", 9),
            foreground="green"
        )
        self.sorting_status.pack(side=tk.RIGHT, padx=(10, 0))

    def setup_manual_tab(self, parent):
        """Setup manual servo control interface"""
        servo_frame = ttk.LabelFrame(parent, text="Servo Control", padding="15")
        servo_frame.pack(fill=tk.BOTH, expand=True)

        # Servo configuration: (num, name, min, max, default)
        servo_configs = [
            (0, "Base", 0, 180, 106),
            (1, "Shoulder", 0, 70, 0), 
            (2, "Elbow", 10, 90, 30),
            (3, "Gripper", 0, 70, 0)
        ]

        # Create servo controls
        for servo_num, name, min_val, max_val, default in servo_configs:
            self.create_servo_control(servo_frame, servo_num, name, min_val, max_val, default)

        # Control buttons
        button_frame = ttk.Frame(servo_frame)
        button_frame.pack(fill=tk.X, pady=(25, 0))

        ttk.Button(
            button_frame, 
            text="Initialize Servos", 
            command=self.initialize_servos,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame, 
            text="Reset All", 
            command=self.reset_servos,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame, 
            text="Test Grab Sequence", 
            command=self.test_grab_sequence,
            width=20
        ).pack(side=tk.LEFT, padx=(0, 10))

        # NEW: Test sorting positions
        ttk.Button(
            button_frame, 
            text="Test Yellow Drop", 
            command=lambda: self.test_drop_position(180),
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame, 
            text="Test Red Drop", 
            command=lambda: self.test_drop_position(0),
            width=15
        ).pack(side=tk.RIGHT)

    def setup_connection_tab(self, parent):
        """Setup connection interface for hardware"""
        conn_frame = ttk.LabelFrame(parent, text="Hardware Connections", padding="15")
        conn_frame.pack(fill=tk.X, pady=(0, 20))

        # Arduino connection section
        arduino_frame = ttk.LabelFrame(conn_frame, text="Arduino Connection", padding="10")
        arduino_frame.pack(fill=tk.X, pady=(0, 15))

        port_frame = ttk.Frame(arduino_frame)
        port_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(port_frame, text="Serial Port:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.port_var = tk.StringVar(master=self.root, value="COM9")
        self.port_combo = ttk.Combobox(
            port_frame, 
            textvariable=self.port_var, 
            width=20,
            state="readonly"
        )
        self.port_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            port_frame, 
            text="🔄 Refresh Ports", 
            command=self.refresh_ports,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.connect_button = ttk.Button(
            port_frame, 
            text="Connect", 
            command=self.toggle_connection,
            width=12
        )
        self.connect_button.pack(side=tk.LEFT)

        # Camera connection section
        camera_frame = ttk.LabelFrame(conn_frame, text="Camera Settings", padding="10")
        camera_frame.pack(fill=tk.X)

        cam_config_frame = ttk.Frame(camera_frame)
        cam_config_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(cam_config_frame, text="Camera Index:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.camera_var = tk.StringVar(master=self.root, value="1")
        ttk.Entry(
            cam_config_frame, 
            textvariable=self.camera_var, 
            width=8,
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Button(
            cam_config_frame, 
            text="Test Camera", 
            command=self.test_camera,
            width=15
        ).pack(side=tk.LEFT)

        # Hardware info section
        info_frame = ttk.LabelFrame(parent, text="System Information", padding="15")
        info_frame.pack(fill=tk.BOTH, expand=True)

        info_text = tk.Text(info_frame, height=12, font=("Courier", 9), state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=info_text.yview)
        info_text.configure(yscrollcommand=info_scrollbar.set)

        info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add system information with sorting details
        system_info = """
Hardware Configuration:
• Servo 0 (Base): 0° - 180° range, controls horizontal rotation
• Servo 1 (Shoulder): 0° - 70° range, controls arm elevation  
• Servo 2 (Elbow): 10° - 90° range, controls forearm angle
• Servo 3 (Gripper): 0° - 70° range, controls gripper open/close

Physical Dimensions:
• Upper arm length (L1): 135mm
• Forearm length (L2): 147mm  
• Base height: 65mm
• Maximum reach: 282mm
• Minimum reach: 12mm

Sorting Configuration (NEW):
• Yellow objects: Drop at base 180° (right side)
• Red objects: Drop at base 0° (left side)
• Auto-sort enabled via checkbox or manual trigger
• Objects must contain "yellow" or "red" in detected class name

Communication Protocol:
• Serial: 9600 baud, 8N1
• Commands: INIT, MOVE <servo> <angle>
• Response: OK/ERROR acknowledgments

Camera Requirements:
• USB camera with 640x480 resolution
• Proper calibration for coordinate transformation
• YOLO model file 'best.pt' in working directory

Safety Features:
• Servo angle limits automatically enforced
• Emergency stop returns to safe position
• Reachability analysis before grab/sort attempts
• Connection status monitoring
• Automatic sequence interruption on errors
        """

        info_text.configure(state=tk.NORMAL)
        info_text.insert(tk.END, system_info)
        info_text.configure(state=tk.DISABLED)

        # Initialize ports list
        self.refresh_ports()

    def create_servo_control(self, parent, servo_num, name, min_val, max_val, default):
        """
        Create servo control widgets (slider, label, entry)

        Args:
            parent: Parent widget
            servo_num: Servo number (0-3)
            name: Display name
            min_val: Minimum angle
            max_val: Maximum angle  
            default: Default angle
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=8)

        # Servo label
        label = ttk.Label(frame, text=f"{name} (Servo {servo_num}):", width=18, font=("Arial", 10, "bold"))
        label.pack(side=tk.LEFT, padx=(0, 15))

        # Range label
        range_label = ttk.Label(frame, text=f"[{min_val}°-{max_val}°]", font=("Arial", 8), foreground="gray")
        range_label.pack(side=tk.LEFT, padx=(0, 10))

        # Slider
        slider = tk.Scale(
            frame, 
            from_=min_val, 
            to=max_val, 
            orient=tk.HORIZONTAL,
            command=lambda val, s=servo_num: self.on_servo_change(s, val),
            length=200,
            font=("Arial", 9)
        )
        slider.set(default)
        slider.pack(side=tk.LEFT, padx=(0, 15), fill=tk.X, expand=True)

        # Value display label
        value_label = ttk.Label(frame, text=f"{default}°", width=6, font=("Arial", 10, "bold"), foreground="blue")
        value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Numeric entry
        entry_var = tk.StringVar(master=self.root, value=str(default))
        entry = ttk.Entry(frame, textvariable=entry_var, width=6, font=("Arial", 10))
        entry.pack(side=tk.LEFT)
        entry.bind('<Return>', lambda e, s=servo_num, v=entry_var: self.on_servo_entry(s, v))

        # Store widget references
        self.servo_sliders[servo_num] = slider
        self.servo_labels[servo_num] = value_label
        self.servo_entries[servo_num] = entry_var

    def toggle_vision(self):
        """Start/stop vision system"""
        if self.vision_running.get():
            self.stop_vision()
        else:
            self.start_vision()

    def start_vision(self):
        """Start vision processing in separate thread"""
        if not self.vision_system:
            messagebox.showerror("Error", "Vision system not available")
            return

        if not self.vision_system.initialize_camera():
            messagebox.showerror("Error", "Failed to initialize camera.\n\nPlease check:\n• Camera is connected\n• Camera index is correct\n• Camera is not used by other applications")
            return

        self.vision_running.set(True)
        self.vision_button.config(text="Stop Vision")

        # Start vision worker thread
        self.vision_thread = threading.Thread(target=self.vision_worker, daemon=True)
        self.vision_thread.start()

        self.status_label.config(text="Vision system started - Camera processing active")
        logger.info("Vision system started")

    def stop_vision(self):
        """Stop vision processing and cleanup"""
        self.vision_running.set(False)
        self.vision_button.config(text="Start Vision")

        # Wait for thread to finish
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2)

        # Cleanup resources
        if self.vision_system:
            self.vision_system.cleanup()
        cv2.destroyAllWindows()

        self.status_label.config(text="Vision system stopped")
        logger.info("Vision system stopped")

    def vision_worker(self):
        """
        Vision processing worker thread
        Runs continuously while vision system is active
        """
        logger.info("Vision worker thread started")

        while self.vision_running.get():
            try:
                # Process camera frame
                frame, objects = self.vision_system.process_frame()

                if frame is not None:
                    # Update frame queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Remove old frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass

                    # Update objects queue
                    try:
                        self.objects_queue.put_nowait(objects)
                    except queue.Full:
                        try:
                            self.objects_queue.get_nowait()
                            self.objects_queue.put_nowait(objects)
                        except queue.Empty:
                            pass

                    # Display frame
                    cv2.imshow("Robotic Arm Vision - Press 'q' to close", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.vision_running.set(False)
                        # Update GUI from main thread
                        self.root.after(0, lambda: self.vision_button.config(text="Start Vision"))

                    # Process auto-grab logic
                    if self.auto_grab_enabled.get() and self.serial_controller.connected:
                        self.process_auto_grab(objects)

                    # Process auto-sort logic
                    if self.auto_sort_enabled.get() and self.serial_controller.connected:
                        self.serial_controller.sort_objects(objects)

            except Exception as e:
                logger.error(f"Vision worker error: {e}")
                time.sleep(0.1)

            # Control frame rate (~30 FPS)
            time.sleep(0.03)

        logger.info("Vision worker thread ended")

    def process_auto_grab(self, objects: List[DetectedObject]):
        """
        Process objects for automatic grabbing

        Args:
            objects: List of detected objects from current frame
        """
        for obj in objects:
            if (obj.name in self.selected_objects and 
                obj.reachable and 
                obj.servo_angles and 
                obj.confidence > 0.5):

                logger.info(f"Auto-grabbing object: {obj.name} (ID: {obj.id}, confidence: {obj.confidence:.2f})")

                base, shoulder, elbow = obj.servo_angles

                # Execute grab in separate thread to avoid blocking vision
                grab_thread = threading.Thread(
                    target=self.serial_controller.grab_sequence,
                    args=(base, shoulder, elbow),
                    daemon=True
                )
                grab_thread.start()

                # Remove from selected to avoid repeated grabs
                self.selected_objects.discard(obj.name)

                # Update GUI
                self.root.after(0, lambda: self.status_label.config(text=f"Grabbed {obj.name} at ({base:.1f}°, {shoulder:.1f}°, {elbow:.1f}°)"))
                break


    def update_display(self):
        """
        Update GUI display with latest information
        Called periodically to refresh object list
        """
        try:
            # Get latest detected objects
            objects = self.objects_queue.get_nowait()
            self.update_object_list(objects)
        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_display)

    def update_object_list(self, objects: List[DetectedObject]):
        """
        Update the object selection listbox with detected objects

        Args:
            objects: List of currently detected objects
        """
        current_objects = set(obj.name for obj in objects if obj.reachable and obj.confidence > 0.5)
        listbox_objects = set(self.object_listbox.get(0, tk.END))

        # Add new objects to listbox
        for obj_name in current_objects - listbox_objects:
            self.object_listbox.insert(tk.END, obj_name)
            logger.debug(f"Added object to list: {obj_name}")

    def on_object_select(self, event):
        """Handle object selection in listbox"""
        selection = self.object_listbox.curselection()
        self.selected_objects = {self.object_listbox.get(i) for i in selection}

        if self.selected_objects:
            self.status_label.config(text=f"Selected objects for auto-grab: {', '.join(self.selected_objects)}")
            logger.info(f"Selected objects: {self.selected_objects}")
        else:
            self.status_label.config(text="No objects selected for auto-grab")

    def clear_object_selections(self):
        """Clear all object selections"""
        self.object_listbox.selection_clear(0, tk.END)
        self.selected_objects.clear()
        self.status_label.config(text="Object selections cleared")
        logger.info("Object selections cleared")

    def grab_selected_objects(self):
        """Manually trigger grab for currently selected objects"""
        if not self.serial_controller.connected:
            messagebox.showwarning("Warning", "Please connect to Arduino first")
            return

        if not self.selected_objects:
            messagebox.showinfo("Info", "No objects selected. Please select objects from the list first.")
            return

        try:
            objects = self.objects_queue.get_nowait()
            grabbed = False

            for obj in objects:
                if (obj.name in self.selected_objects and 
                    obj.reachable and 
                    obj.servo_angles and 
                    obj.confidence > 0.5):

                    logger.info(f"Manual grab triggered for: {obj.name}")
                    base, shoulder, elbow = obj.servo_angles

                    # Execute grab sequence
                    grab_thread = threading.Thread(
                        target=self.serial_controller.grab_sequence,
                        args=(base, shoulder, elbow),
                        daemon=True
                    )
                    grab_thread.start()

                    self.status_label.config(text=f"Grabbing {obj.name}...")
                    grabbed = True
                    break

            if not grabbed:
                messagebox.showinfo("Info", "No reachable objects found matching your selection")

        except queue.Empty:
            messagebox.showinfo("Info", "No objects currently detected")

    def manual_sort_objects(self):
        """NEW: Manually trigger sorting for currently detected objects"""
        if not self.serial_controller.connected:
            messagebox.showwarning("Warning", "Please connect to Arduino first")
            return

        try:
            objects = self.objects_queue.get_nowait()
            if not objects:
                messagebox.showinfo("Info", "No objects currently detected")
                return

            # Filter for sortable objects
            sortable_objects = []
            for obj in objects:
                if obj.reachable and obj.servo_angles and obj.confidence > 0.5:
                    obj_name_lower = obj.name.lower()
                    if "yellow" in obj_name_lower or "red" in obj_name_lower:
                        sortable_objects.append(obj)

            if not sortable_objects:
                messagebox.showinfo("Info", "No sortable objects detected\n\nLooking for objects with 'yellow' or 'red' in their names that are reachable")
                return

            # Confirm sorting operation
            sort_summary = []
            for obj in sortable_objects:
                color = "yellow" if "yellow" in obj.name.lower() else "red"
                drop_zone = "180°" if color == "yellow" else "0°"
                sort_summary.append(f"• {obj.name} → {drop_zone}")

            response = messagebox.askyesno(
                "Confirm Sorting Operation",
                f"Sort {len(sortable_objects)} objects?\n\n" + "\n".join(sort_summary)
            )

            if response:
                self.status_label.config(text="Starting sorting operation...")
                if self.sorting_status:
                    self.sorting_status.config(text="Sorting Status: Running", foreground="orange")

                # Execute sorting in separate thread
                sort_thread = threading.Thread(
                    target=self._execute_sorting_thread,
                    args=(sortable_objects,),
                    daemon=True
                )
                sort_thread.start()

        except queue.Empty:
            messagebox.showinfo("Info", "No objects currently detected")

    def _execute_sorting_thread(self, objects):
        """Execute sorting in background thread"""
        try:
            sorted_count = self.serial_controller.sort_objects(objects)

            # Update GUI from main thread
            self.root.after(0, lambda: self._sorting_complete(sorted_count))

        except Exception as e:
            logger.error(f"Sorting thread error: {e}")
            self.root.after(0, lambda: self._sorting_error(str(e)))

    def _sorting_complete(self, sorted_count):
        """Handle sorting completion (called from main thread)"""
        total = sorted_count["yellow"] + sorted_count["red"]
        message = f"Sorting completed: {sorted_count['yellow']} yellow, {sorted_count['red']} red objects sorted"

        self.status_label.config(text=message)
        if self.sorting_status:
            self.sorting_status.config(text="Sorting Status: Complete", foreground="green")

        messagebox.showinfo("Sorting Complete", f"{message}\n\n{sorted_count['other']} objects were not sortable")

    def _sorting_error(self, error_msg):
        """Handle sorting error (called from main thread)"""
        self.status_label.config(text="Sorting failed - check log for details")
        if self.sorting_status:
            self.sorting_status.config(text="Sorting Status: Error", foreground="red")
        messagebox.showerror("Sorting Error", f"Sorting operation failed:\n\n{error_msg}")

    def emergency_stop(self):
        """Emergency stop - immediately return servos to safe position"""
        if self.serial_controller.connected:
            self.serial_controller.send_command("INIT")
            self.status_label.config(text="EMERGENCY STOP - Servos returned to safe position")
            logger.warning("Emergency stop executed")
            messagebox.showinfo("Emergency Stop", "All servos have been returned to safe position")
        else:
            messagebox.showwarning("Warning", "Cannot execute emergency stop: Arduino not connected")

    def on_servo_change(self, servo_num, value):
        """
        Handle servo slider value change

        Args:
            servo_num: Servo number (0-3)
            value: New angle value from slider
        """
        angle = int(float(value))

        # Update display
        self.servo_labels[servo_num].config(text=f"{angle}°")
        self.servo_entries[servo_num].set(str(angle))

        # Send command to Arduino
        if self.serial_controller.connected:
            success = self.serial_controller.move_servo(servo_num, angle)
            if success:
                self.status_label.config(text=f"Servo {servo_num} moved to {angle}°")
            else:
                self.status_label.config(text=f"Failed to move servo {servo_num}")

    def on_servo_entry(self, servo_num, var):
        """
        Handle servo numeric entry submission

        Args:
            servo_num: Servo number (0-3)
            var: StringVar containing entered value
        """
        try:
            angle = int(var.get())

            # Validate angle limits
            min_angle, max_angle = self.serial_controller.servo_limits[servo_num]
            if angle < min_angle or angle > max_angle:
                messagebox.showerror("Invalid Angle", f"Angle must be between {min_angle}° and {max_angle}°")
                var.set(str(self.servo_sliders[servo_num].get()))
                return

            # Update slider and display
            self.servo_sliders[servo_num].set(angle)
            self.servo_labels[servo_num].config(text=f"{angle}°")

            # Send command to Arduino
            if self.serial_controller.connected:
                success = self.serial_controller.move_servo(servo_num, angle)
                if success:
                    self.status_label.config(text=f"Servo {servo_num} set to {angle}°")
                else:
                    self.status_label.config(text=f"Failed to set servo {servo_num}")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer angle")
            var.set(str(self.servo_sliders[servo_num].get()))

    def toggle_connection(self):
        """Toggle Arduino connection"""
        if self.serial_controller.connected:
            self.disconnect_arduino()
        else:
            self.connect_arduino()

    def connect_arduino(self):
        """Establish connection to Arduino"""
        port = self.port_var.get()

        if port == 'No ports found':
            messagebox.showerror("Error", "No serial ports available")
            return

        if self.serial_controller.connect(port):
            self.connect_button.config(text="Disconnect")
            self.status_label.config(text=f"Connected to Arduino on {port}")

            # Initialize servos to safe position
            self.root.after(1000, self.initialize_servos)  # Delay to allow Arduino to initialize

        else:
            messagebox.showerror("Connection Error", f"Failed to connect to {port}\n\nPlease check:\n• Arduino is connected\n• Correct port is selected\n• Arduino is not used by other programs")

    def disconnect_arduino(self):
        """Disconnect from Arduino"""
        self.serial_controller.disconnect()
        self.connect_button.config(text="Connect")
        self.status_label.config(text="Disconnected from Arduino")

    def refresh_ports(self):
        """Refresh available serial ports list"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports if ports else ['No ports found']

        # Update selection if current port not available
        current_port = self.port_var.get()
        if ports:
            if current_port not in ports:
                self.port_var.set(ports[0])
        else:
            self.port_var.set('No ports found')

        logger.info(f"Found {len(ports)} serial ports: {ports}")

    def test_camera(self):
        """Test camera connectivity and display test frame"""
        try:
            camera_index = int(self.camera_var.get())
            cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Camera Test - Press any key to close", frame)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Camera Test - Press any key to close")
                    messagebox.showinfo("Success", f"Camera {camera_index} test successful!\n\nResolution: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    messagebox.showerror("Error", f"Camera {camera_index} opened but failed to capture frame")
                cap.release()
            else:
                messagebox.showerror("Error", f"Cannot open camera {camera_index}\n\nPlease check:\n• Camera is connected\n• Camera index is correct\n• Camera permissions are granted")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid camera index (usually 0 or 1)")
        except Exception as e:
            messagebox.showerror("Error", f"Camera test failed: {str(e)}")

    def initialize_servos(self):
        """Initialize servos to safe starting positions"""
        if self.serial_controller.connected:
            success = self.serial_controller.send_command("INIT")
            if success:
                self.status_label.config(text="Servos initialized to safe position")
                logger.info("Servos initialized successfully")
            else:
                self.status_label.config(text="Failed to initialize servos")
                logger.error("Servo initialization failed")

            time.sleep(1)  # Allow servos to reach position
        else:
            messagebox.showwarning("Warning", "Please connect to Arduino first")

    def reset_servos(self):
        """Reset all servos to default positions and update GUI"""
        if not self.serial_controller.connected:
            messagebox.showwarning("Warning", "Please connect to Arduino first")
            return

        # Default positions
        defaults = {0: 90, 1: 0, 2: 30, 3: 0}

        for servo_num, angle in defaults.items():
            # Update GUI elements
            self.servo_sliders[servo_num].set(angle)
            self.servo_labels[servo_num].config(text=f"{angle}°")
            self.servo_entries[servo_num].set(str(angle))

            # Send command to Arduino
            if self.serial_controller.move_servo(servo_num, angle):
                time.sleep(0.1)  # Small delay between commands
            else:
                logger.error(f"Failed to reset servo {servo_num}")

        self.status_label.config(text="All servos reset to default positions")
        logger.info("All servos reset to defaults")

    def test_grab_sequence(self):
        """Test grab sequence using current servo positions"""
        if not self.serial_controller.connected:
            messagebox.showwarning("Warning", "Please connect to Arduino first")
            return

        # Get current positions from sliders
        base = self.servo_sliders[0].get()
        shoulder = self.servo_sliders[1].get()
        elbow = self.servo_sliders[2].get()

        # Confirm with user
        response = messagebox.askyesno(
            "Test Grab Sequence",
            f"Execute grab sequence with current positions?\n\nBase: {base}°\nShoulder: {shoulder}°\nElbow: {elbow}°\n\nThis will move the arm!"
        )

        if response:
            self.status_label.config(text="Executing test grab sequence...")

            # Execute in separate thread to avoid blocking GUI
            test_thread = threading.Thread(
                target=self.serial_controller.grab_sequence,
                args=(base, shoulder, elbow),
                daemon=True
            )
            test_thread.start()

            logger.info(f"Test grab sequence executed: Base={base}°, Shoulder={shoulder}°, Elbow={elbow}°")

    def test_drop_position(self, base_angle):
        """NEW: Test drop position for sorting"""
        if not self.serial_controller.connected:
            messagebox.showwarning("Warning", "Please connect to Arduino first")
            return

        color = "Yellow" if base_angle == 180 else "Red"
        response = messagebox.askyesno(
            f"Test {color} Drop Position",
            f"Test {color.lower()} drop zone at base {base_angle}°?\n\nThis will move the arm to the drop position."
        )

        if response:
            self.status_label.config(text=f"Testing {color.lower()} drop position...")

            # Test sequence: move to drop position and back
            test_thread = threading.Thread(
                target=self._test_drop_position_thread,
                args=(base_angle, color),
                daemon=True
            )
            test_thread.start()

    def _test_drop_position_thread(self, base_angle, color):
        """Execute drop position test in background thread"""
        try:
            commands = [
                ("INIT", 1.0),
                (f"MOVE 0 {base_angle}", 0.5),
                (f"MOVE 1 {self.serial_controller.drop_shoulder_angle}", 0.5),
                (f"MOVE 2 {self.serial_controller.drop_elbow_angle}", 0.5),
                ("Wait at drop position", 2.0),
                ("INIT", 1.0)
            ]

            for command, delay in commands:
                if command.startswith("Wait"):
                    time.sleep(delay)
                else:
                    self.serial_controller.send_command(command)
                    time.sleep(delay)

            self.root.after(0, lambda: self.status_label.config(text=f"{color} drop position test completed"))

        except Exception as e:
            logger.error(f"Drop position test error: {e}")
            self.root.after(0, lambda: self.status_label.config(text="Drop position test failed"))

    def on_closing(self):
        """Handle application exit and cleanup"""
        logger.info("Application shutting down...")

        # Stop vision system
        if self.vision_running.get():
            self.stop_vision()

        # Disconnect Arduino
        if self.serial_controller.connected:
            self.serial_controller.disconnect()

        # Destroy GUI
        self.root.destroy()

        logger.info("Application shutdown complete")

    def run(self):
        """Start the GUI application main loop"""
        # Set window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info("Starting GUI main loop")

        # Start main loop
        self.root.mainloop()


def main():
    """
    Main entry point for the application

    Handles application startup, error catching, and graceful shutdown
    """
    logger.info("=" * 60)
    logger.info("Starting Advanced Robotic Arm Controller with Sorting")
    logger.info("=" * 60)
    logger.info(f"Python version: {tk.TkVersion}")
    logger.info(f"OpenCV available: {cv2 is not None}")
    logger.info(f"YOLO available: {YOLO_AVAILABLE}")


    try:
        # Create and run application
        logger.info("Creating GUI application...")
        app = RoboticArmGUI()
        logger.info("GUI created successfully, starting main loop...")
        app.run()

    except KeyboardInterrupt:
        logger.info("Application terminated by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"Unexpected application error: {e}")
        logger.exception("Full error traceback:")

        # Show error dialog if possible
        try:
            messagebox.showerror("Critical Error", f"Application encountered an error:\n\n{str(e)}\n\nSee log file for details.")
        except:
            pass

    finally:
        logger.info("Application cleanup completed")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
